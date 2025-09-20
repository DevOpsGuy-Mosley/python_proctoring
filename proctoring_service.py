#!/usr/bin/env python3
"""
Microservice Python pour le Proctoring IA
Plateforme Sélect - Surveillance en temps réel
"""

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import numpy as np
import base64
import json
import mysql.connector
from datetime import datetime
import logging
import os
from typing import Dict, List, Any
import uuid
import pyaudio
import wave
import threading
import time

# Configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'select_proctoring_secret_2024'

# Configuration CORS explicite
from flask_cors import CORS
CORS(app, origins="*")

socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration base de données
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASS', ''),
    'database': os.getenv('DB_NAME', 'select_recruitment'),
    'charset': 'utf8mb4'
}

# Configuration OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Sessions actives
active_sessions: Dict[str, Dict] = {}

# Configuration audio
AUDIO_CONFIG = {
    'CHUNK': 1024,
    'FORMAT': pyaudio.paInt16,
    'CHANNELS': 1,
    'RATE': 44100,
    'THRESHOLD': 500,  # Seuil de détection de voix
    'SILENCE_DURATION': 2.0  # Durée de silence avant alerte
}

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceDetector:
    """Détecteur de voix en temps réel"""
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_listening = False
        self.voice_detected = False
        self.silence_start = None
        
    def start_listening(self, session_id: str):
        """Démarrer l'écoute audio pour une session"""
        def audio_worker():
            try:
                stream = self.audio.open(
                    format=AUDIO_CONFIG['FORMAT'],
                    channels=AUDIO_CONFIG['CHANNELS'],
                    rate=AUDIO_CONFIG['RATE'],
                    input=True,
                    frames_per_buffer=AUDIO_CONFIG['CHUNK']
                )
                
                self.is_listening = True
                logger.info(f"Démarrage écoute audio pour session {session_id}")
                
                while self.is_listening:
                    data = stream.read(AUDIO_CONFIG['CHUNK'])
                    audio_level = np.frombuffer(data, dtype=np.int16).max()
                    
                    if audio_level > AUDIO_CONFIG['THRESHOLD']:
                        if not self.voice_detected:
                            self.voice_detected = True
                            self.silence_start = None
                            logger.info(f"Voix détectée dans session {session_id}")
                            
                            # Envoyer alerte de voix
                            socketio.emit('voice_detected', {
                                'session_id': session_id,
                                'audio_level': audio_level,
                                'timestamp': datetime.now().isoformat()
                            }, room=session_id)
                    else:
                        if self.voice_detected:
                            if self.silence_start is None:
                                self.silence_start = time.time()
                            elif time.time() - self.silence_start > AUDIO_CONFIG['SILENCE_DURATION']:
                                self.voice_detected = False
                                self.silence_start = None
                                logger.info(f"Silence détecté dans session {session_id}")
                
                stream.stop_stream()
                stream.close()
                
            except Exception as e:
                logger.error(f"Erreur écoute audio: {str(e)}")
        
        thread = threading.Thread(target=audio_worker)
        thread.daemon = True
        thread.start()
    
    def stop_listening(self):
        """Arrêter l'écoute audio"""
        self.is_listening = False

class ProctoringAnalyzer:
    """Analyseur de proctoring avec OpenCV"""
    
    def __init__(self):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.voice_detector = VoiceDetector()
    
    def analyze_frame(self, frame_data: str, session_id: str = None) -> Dict[str, Any]:
        """Analyser une frame vidéo pour détecter les anomalies"""
        try:
            # Décoder l'image base64
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            image_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"Impossible de décoder l'image pour session {session_id}")
                return {"error": "Impossible de décoder l'image"}
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Détecter les visages
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            
            result = {
                "faces_detected": len(faces),
                "anomalies": [],
                "frame_analyzed": True,
                "timestamp": datetime.now().isoformat(),
                "should_warn": False,
                "should_fail": False
            }
            
            # Analyser les anomalies
            if len(faces) == 0:
                # Aucun visage détecté
                result["anomalies"].append({
                    "type": "no_face",
                    "severity": "high",
                    "description": "Aucun visage détecté dans la frame - Sortie de cadre détectée",
                    "confidence": 0.9
                })
                result["should_warn"] = True
                result["face_detected"] = False
                logger.warning(f"Sortie de cadre détectée pour session {session_id}")
                
            elif len(faces) > 1:
                # Plusieurs visages détectés
                result["anomalies"].append({
                    "type": "multiple_faces",
                    "severity": "high",
                    "description": f"{len(faces)} visages détectés - Personne supplémentaire détectée",
                    "confidence": 0.95
                })
                result["should_warn"] = True
                result["face_detected"] = True
                logger.warning(f"Plusieurs visages détectés ({len(faces)}) pour session {session_id}")
                
            else:
                # Un seul visage détecté - analyser les yeux
                result["face_detected"] = True
                face = faces[0]
                x, y, w, h = face
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) < 2:
                    result["anomalies"].append({
                        "type": "looking_away",
                        "severity": "medium",
                        "description": "Regard détourné ou yeux non détectés",
                        "confidence": 0.7
                    })
                    result["should_warn"] = True
                    logger.warning(f"Regard détourné détecté pour session {session_id}")
                
                # Vérifier la taille du visage (trop loin/trop près)
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                if face_ratio < 0.05:
                    result["anomalies"].append({
                        "type": "face_too_small",
                        "severity": "medium",
                        "description": "Visage trop petit (trop loin de la caméra)",
                        "confidence": 0.6
                    })
                elif face_ratio > 0.3:
                    result["anomalies"].append({
                        "type": "face_too_large",
                        "severity": "low",
                        "description": "Visage trop grand (trop près de la caméra)",
                        "confidence": 0.5
                    })
            
            # Le nouveau système de surveillance 5/5 est géré côté frontend
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse frame: {str(e)}")
            return {"error": f"Erreur d'analyse: {str(e)}"}
    
    def detect_audio_anomaly(self, audio_level: float) -> Dict[str, Any]:
        """Détecter les anomalies audio (simulation)"""
        if audio_level > 0.8:
            return {
                "type": "audio_anomaly",
                "severity": "medium",
                "description": "Niveau audio élevé détecté",
                "confidence": 0.7
            }
        return None

class DatabaseManager:
    """Gestionnaire de base de données MySQL"""
    
    def __init__(self):
        self.config = DB_CONFIG
    
    def get_connection(self):
        """Obtenir une connexion à la base de données"""
        return mysql.connector.connect(**self.config)
    
    def save_proctoring_alert(self, session_id: str, alert_data: Dict) -> bool:
        """Sauvegarder une alerte de proctoring"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO proctoring_alerts 
                (id, session_id, alert_type, severity, description, timestamp, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                str(uuid.uuid4()),
                session_id,
                alert_data['type'],
                alert_data['severity'],
                alert_data['description'],
                datetime.now(),
                alert_data.get('confidence', 0.0)
            )
            
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde alerte: {str(e)}")
            return False
    
    def save_violation(self, session_id: str, violation_data: Dict) -> bool:
        """Sauvegarder une violation"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = """
                INSERT INTO violations 
                (id, session_id, violation_type, severity, description, timestamp, user_action)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                str(uuid.uuid4()),
                session_id,
                violation_data['type'],
                violation_data['severity'],
                violation_data['description'],
                datetime.now(),
                violation_data.get('user_action', '')
            )
            
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde violation: {str(e)}")
            return False

# Initialiser les composants
analyzer = ProctoringAnalyzer()
db_manager = DatabaseManager()

# Routes API
@app.route('/health', methods=['GET'])
def health_check():
    """Vérification de santé du service"""
    return jsonify({
        "status": "healthy",
        "service": "select_proctoring",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions)
    })

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """API pour analyser une frame vidéo"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        frame_data = data.get('frame')
        
        if not session_id or not frame_data:
            return jsonify({"error": "session_id et frame requis"}), 400
        
        # Analyser la frame
        result = analyzer.analyze_frame(frame_data)
        
        # Sauvegarder les alertes
        if result.get('anomalies'):
            for anomaly in result['anomalies']:
                db_manager.save_proctoring_alert(session_id, anomaly)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur analyse frame API: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Events WebSocket
@socketio.on('connect')
def handle_connect():
    """Client connecté"""
    logger.info(f"Client connecté: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Client déconnecté"""
    logger.info(f"Client déconnecté: {request.sid}")
    
    # Nettoyer les sessions
    for session_id, session_data in list(active_sessions.items()):
        if session_data.get('socket_id') == request.sid:
            del active_sessions[session_id]

@socketio.on('join_test_session')
def handle_join_session(data):
    """Rejoindre une session de test"""
    session_id = data.get('session_id')
    
    if session_id:
        active_sessions[session_id] = {
            'socket_id': request.sid,
            'start_time': datetime.now(),
            'anomaly_count': 0,
            'violation_count': 0,
            'warning_count': 0,
            'voice_detected': False,
            'test_failed': False
        }
        
        # Démarrer la détection de voix
        analyzer.voice_detector.start_listening(session_id)
        
        join_room(session_id)
        emit('session_joined', {
            'status': 'success', 
            'session_id': session_id,
            'warning_system': {
                'max_warnings': 3,
                'current_warnings': 0
            }
        })
        logger.info(f"Session {session_id} rejointe par {request.sid}")

@socketio.on('video_frame')
def handle_video_frame(data):
    """Traiter une frame vidéo"""
    session_id = data.get('session_id')
    frame_data = data.get('frame')
    
    if not session_id:
        logger.warning("Frame reçue sans session_id")
        return
        
    if session_id not in active_sessions:
        logger.warning(f"Frame reçue pour session inexistante: {session_id}")
        return
    
    # Vérifier si le test a déjà échoué
    if active_sessions[session_id].get('test_failed', False):
        return
    
    logger.info(f"Frame reçue pour session {session_id}, taille: {len(frame_data) if frame_data else 0}")
    
    # Analyser la frame
    analysis = analyzer.analyze_frame(frame_data, session_id)
    
    if analysis.get('anomalies'):
        active_sessions[session_id]['anomaly_count'] += len(analysis['anomalies'])
        
        # Sauvegarder les alertes
        for anomaly in analysis['anomalies']:
            db_manager.save_proctoring_alert(session_id, anomaly)
        
        # Gérer les alertes immédiatement (système de surveillance 5/5)
        if analysis.get('should_warn'):
            # Envoyer l'alerte immédiatement pour chaque anomalie
            for anomaly in analysis['anomalies']:
                emit('proctoring_alert', {
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'description': anomaly['description'],
                    'confidence': anomaly['confidence'],
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
            
            logger.info(f"Alerte envoyée pour session {session_id}: {len(analysis['anomalies'])} anomalies")
        
        # Envoyer un signal si un visage est détecté (pour arrêter le décompte)
        if analysis.get('face_detected'):
            emit('face_detected', {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id
            }, room=session_id)
        
        # Le système d'échec est maintenant géré côté frontend (surveillance 5/5)
        
        logger.info(f"Analyse proctoring pour session {session_id}: {len(analysis['anomalies'])} anomalies")

@socketio.on('test_violation')
def handle_test_violation(data):
    """Traiter une violation de test"""
    session_id = data.get('session_id')
    violation_type = data.get('type')
    user_action = data.get('user_action', '')
    
    if session_id and session_id in active_sessions:
        violation_data = {
            'type': violation_type,
            'severity': 'high' if violation_type in ['dev_tools', 'tab_switch'] else 'medium',
            'description': f"Violation détectée: {violation_type}",
            'user_action': user_action
        }
        
        # Sauvegarder la violation
        db_manager.save_violation(session_id, violation_data)
        
        active_sessions[session_id]['violation_count'] += 1
        
        # Envoyer l'alerte au client
        emit('violation_recorded', {
            'type': violation_type,
            'total_violations': active_sessions[session_id]['violation_count'],
            'timestamp': datetime.now().isoformat()
        }, room=session_id)
        
        logger.info(f"Violation enregistrée pour session {session_id}: {violation_type}")

@socketio.on('audio_anomaly')
def handle_audio_anomaly(data):
    """Traiter une anomalie audio"""
    session_id = data.get('session_id')
    audio_level = data.get('audio_level', 0.0)
    
    if session_id and session_id in active_sessions:
        # Vérifier si le test a déjà échoué
        if active_sessions[session_id].get('test_failed', False):
            return
            
        anomaly = analyzer.detect_audio_anomaly(audio_level)
        
        if anomaly:
            # Marquer la voix comme détectée
            active_sessions[session_id]['voice_detected'] = True
            
            # Ajouter un avertissement pour la voix
            active_sessions[session_id]['warning_count'] += 1
            warning_count = active_sessions[session_id]['warning_count']
            
            # Sauvegarder l'alerte
            db_manager.save_proctoring_alert(session_id, anomaly)
            
            # Vérifier si on atteint 3 avertissements
            if warning_count >= 3:
                active_sessions[session_id]['test_failed'] = True
                
                # Marquer le test comme échoué dans la base de données
                try:
                    conn = db_manager.get_connection()
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE test_sessions SET status = 'failed', end_time = NOW() WHERE id = %s",
                        (session_id,)
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                except Exception as e:
                    logger.error(f"Erreur mise à jour statut test: {str(e)}")
                
                emit('test_failed', {
                    'reason': 'Trop d\'avertissements - Voix détectée',
                    'warning_count': warning_count,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
                
                logger.info(f"Test échoué automatiquement pour session {session_id} - Voix détectée (3 avertissements)")
            else:
                emit('proctoring_warning', {
                    'anomalies': [anomaly],
                    'warning_count': warning_count,
                    'max_warnings': 3,
                    'remaining_warnings': 3 - warning_count,
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
                
                logger.info(f"Avertissement {warning_count}/3 pour session {session_id} - Voix détectée")
            
            emit('proctoring_alert', {
                'anomalies': [anomaly],
                'warning_count': warning_count,
                'test_failed': active_sessions[session_id]['test_failed'],
                'timestamp': datetime.now().isoformat()
            }, room=session_id)

@socketio.on('session_end')
def handle_session_end(data):
    """Fin de session"""
    session_id = data.get('session_id')
    
    if session_id and session_id in active_sessions:
        session_data = active_sessions[session_id]
        
        # Calculer les statistiques finales
        total_anomalies = session_data['anomaly_count']
        total_violations = session_data['violation_count']
        
        # Calculer le score de crédibilité
        credibility_score = max(0, 100 - (total_anomalies * 5) - (total_violations * 10))
        
        emit('session_statistics', {
            'total_anomalies': total_anomalies,
            'total_violations': total_violations,
            'credibility_score': credibility_score,
            'session_duration': (datetime.now() - session_data['start_time']).total_seconds()
        }, room=session_id)
        
        # Nettoyer la session
        del active_sessions[session_id]
        leave_room(session_id)
        
        logger.info(f"Session {session_id} terminée - Anomalies: {total_anomalies}, Violations: {total_violations}")

if __name__ == '__main__':
    logger.info("Démarrage du service de proctoring IA")
    logger.info(f"Configuration DB: {DB_CONFIG['host']}:{DB_CONFIG['database']}")
    
    # Vérifier la connexion à la base de données
    try:
        conn = db_manager.get_connection()
        conn.close()
        logger.info("Connexion à la base de données OK")
    except Exception as e:
        logger.error(f"Erreur connexion DB: {str(e)}")
    
    # Démarrer le serveur
    socketio.run(app, host='0.0.0.0', port=8001, debug=True)
