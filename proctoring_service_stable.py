#!/usr/bin/env python3
"""
Microservice Python pour le Proctoring IA - Version Stable
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

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProctoringAnalyzer:
    """Analyseur de proctoring avec OpenCV - Version stable"""
    
    def __init__(self):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        # Compteurs pour éviter les alertes trop fréquentes
        self.last_no_face_alert = {}
        self.no_face_count = {}
        self.ALERT_COOLDOWN = 10  # 10 secondes entre les alertes
        self.NO_FACE_THRESHOLD = 5  # 5 frames consécutives sans visage avant alerte
    
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
            
            # Détecter les visages avec paramètres plus stables
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,  # Plus strict pour éviter les faux positifs
                minSize=(30, 30),  # Taille minimale du visage
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            result = {
                "faces_detected": len(faces),
                "anomalies": [],
                "frame_analyzed": True,
                "timestamp": datetime.now().isoformat(),
                "should_warn": False,
                "should_fail": False
            }
            
            current_time = time.time()
            
            # Analyser les anomalies
            if len(faces) == 0:
                # Aucun visage détecté - gérer le compteur
                if session_id not in self.no_face_count:
                    self.no_face_count[session_id] = 0
                
                self.no_face_count[session_id] += 1
                result["face_detected"] = False
                
                # Envoyer l'alerte seulement après plusieurs frames sans visage
                if self.no_face_count[session_id] >= self.NO_FACE_THRESHOLD:
                    # Vérifier le cooldown pour éviter les alertes trop fréquentes
                    if (session_id not in self.last_no_face_alert or 
                        current_time - self.last_no_face_alert[session_id] > self.ALERT_COOLDOWN):
                        
                        result["anomalies"].append({
                            "type": "no_face",
                            "severity": "high",
                            "description": "Aucun visage détecté - Sortie de cadre détectée",
                            "confidence": 0.9
                        })
                        result["should_warn"] = True
                        self.last_no_face_alert[session_id] = current_time
                        logger.warning(f"Sortie de cadre détectée pour session {session_id}")
                
            elif len(faces) > 1:
                # Plusieurs visages détectés
                result["face_detected"] = True
                result["anomalies"].append({
                    "type": "multiple_faces",
                    "severity": "high",
                    "description": f"{len(faces)} visages détectés - Personne supplémentaire détectée",
                    "confidence": 0.95
                })
                result["should_warn"] = True
                logger.warning(f"Plusieurs visages détectés ({len(faces)}) pour session {session_id}")
                
                # Réinitialiser le compteur no_face
                if session_id in self.no_face_count:
                    self.no_face_count[session_id] = 0
                    
            else:
                # Un seul visage détecté - analyser les yeux
                result["face_detected"] = True
                
                # Réinitialiser le compteur no_face
                if session_id in self.no_face_count:
                    self.no_face_count[session_id] = 0
                
                face = faces[0]
                x, y, w, h = face
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
                
                # Vérifier la taille du visage (trop loin/trop près)
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                # Seuils plus réalistes
                if face_ratio < 0.03:  # Visage vraiment très petit
                    result["anomalies"].append({
                        "type": "face_too_small",
                        "severity": "medium",
                        "description": "Visage trop petit (trop loin de la caméra)",
                        "confidence": 0.6
                    })
                    result["should_warn"] = True
                elif face_ratio > 0.4:  # Visage vraiment très grand
                    result["anomalies"].append({
                        "type": "face_too_large",
                        "severity": "low",
                        "description": "Visage trop grand (trop près de la caméra)",
                        "confidence": 0.5
                    })
                
                # Détection de regard détourné plus stricte
                if len(eyes) < 2 and face_ratio > 0.05:  # Seulement si le visage n'est pas trop petit
                    result["anomalies"].append({
                        "type": "looking_away",
                        "severity": "medium",
                        "description": "Regard détourné ou yeux non détectés",
                        "confidence": 0.7
                    })
                    result["should_warn"] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse frame: {str(e)}")
            return {"error": f"Erreur d'analyse: {str(e)}"}

class DatabaseManager:
    """Gestionnaire de base de données MySQL"""
    
    def __init__(self):
        self.config = DB_CONFIG
        self.db_available = False
        
        # Tester la connexion au démarrage
        try:
            conn = mysql.connector.connect(**self.config)
            conn.close()
            self.db_available = True
            logger.info("Connexion à la base de données OK")
        except Exception as e:
            logger.warning(f"Base de données non disponible: {str(e)}")
            self.db_available = False
    
    def get_connection(self):
        """Obtenir une connexion à la base de données"""
        if not self.db_available:
            raise Exception("Base de données non disponible")
        return mysql.connector.connect(**self.config)
    
    def save_proctoring_alert(self, session_id: str, alert_data: Dict) -> bool:
        """Sauvegarder une alerte de proctoring"""
        if not self.db_available:
            return False
            
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
        if not self.db_available:
            return False
            
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
@app.route('/', methods=['GET'])
def root():
    """Route racine pour éviter les erreurs 404"""
    return jsonify({
        "service": "select_proctoring",
        "status": "running",
        "version": "1.0.1",
        "timestamp": datetime.now().isoformat()
    })

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
        result = analyzer.analyze_frame(frame_data, session_id)
        
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
            # Nettoyer les compteurs
            if session_id in analyzer.no_face_count:
                del analyzer.no_face_count[session_id]
            if session_id in analyzer.last_no_face_alert:
                del analyzer.last_no_face_alert[session_id]

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
            'test_failed': False
        }
        
        # Initialiser les compteurs pour cette session
        analyzer.no_face_count[session_id] = 0
        
        join_room(session_id)
        emit('session_joined', {
            'status': 'success', 
            'session_id': session_id,
            'warning_system': {
                'max_warnings': 5,  # Système 5/5
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
        return
        
    if session_id not in active_sessions:
        return
    
    # Vérifier si le test a déjà échoué
    if active_sessions[session_id].get('test_failed', False):
        return
    
    try:
        # Analyser la frame
        analysis = analyzer.analyze_frame(frame_data, session_id)
        
        # Toujours envoyer un signal si un visage est détecté (pour arrêter le décompte)
        if analysis.get('face_detected'):
            emit('face_detected', {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'faces_count': analysis.get('faces_detected', 0)
            }, room=session_id)
        
        # Envoyer les alertes seulement s'il y en a
        if analysis.get('anomalies') and analysis.get('should_warn'):
            active_sessions[session_id]['anomaly_count'] += len(analysis['anomalies'])
            
            # Sauvegarder les alertes (optionnel)
            for anomaly in analysis['anomalies']:
                db_manager.save_proctoring_alert(session_id, anomaly)
            
            # Envoyer l'alerte pour chaque anomalie
            for anomaly in analysis['anomalies']:
                emit('proctoring_alert', {
                    'type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'description': anomaly['description'],
                    'confidence': anomaly['confidence'],
                    'timestamp': datetime.now().isoformat()
                }, room=session_id)
            
            logger.info(f"Alerte envoyée pour session {session_id}: {anomaly['type']}")
        
    except Exception as e:
        logger.error(f"Erreur analyse frame pour session {session_id}: {str(e)}")

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
        
        # Nettoyer les compteurs
        if session_id in analyzer.no_face_count:
            del analyzer.no_face_count[session_id]
        if session_id in analyzer.last_no_face_alert:
            del analyzer.last_no_face_alert[session_id]
        
        logger.info(f"Session {session_id} terminée - Anomalies: {total_anomalies}, Violations: {total_violations}")

if __name__ == '__main__':
    logger.info("Démarrage du service de proctoring IA - Version Stable")
    logger.info(f"Configuration DB: {DB_CONFIG['host']}:{DB_CONFIG['database']}")
    
    # Démarrer le serveur
    socketio.run(app, host='0.0.0.0', port=8001, debug=True)
