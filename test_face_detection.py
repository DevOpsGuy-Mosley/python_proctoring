#!/usr/bin/env python3
"""
Test de détection de visage pour le service de proctoring
"""

import requests
import json
import base64
import cv2
import numpy as np

def create_test_image():
    """Créer une image de test avec un visage simulé"""
    # Créer une image blanche 640x480
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Dessiner un cercle pour simuler un visage
    cv2.circle(img, (320, 240), 100, (200, 200, 200), -1)
    
    # Dessiner des yeux
    cv2.circle(img, (300, 220), 10, (0, 0, 0), -1)
    cv2.circle(img, (340, 220), 10, (0, 0, 0), -1)
    
    # Dessiner une bouche
    cv2.ellipse(img, (320, 260), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Encoder en base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"

def test_face_detection():
    """Tester la détection de visage"""
    print("🧪 Test de détection de visage...")
    
    # URL du service
    url = "https://python-proctoring.onrender.com/api/analyze-frame"
    
    # Créer une image de test
    test_image = create_test_image()
    
    # Données à envoyer
    data = {
        "session_id": "test_session_123",
        "frame": test_image,
        "timestamp": "2025-09-20T10:55:00"
    }
    
    # Headers
    headers = {
        "Content-Type": "application/json",
        "Origin": "https://p6-groupeb.com"
    }
    
    try:
        print(f"📤 Envoi vers: {url}")
        print(f"📊 Taille image: {len(test_image)} caractères")
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        print(f"📥 Status Code: {response.status_code}")
        print(f"📥 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Réponse: {json.dumps(result, indent=2)}")
            
            # Vérifier les résultats
            if result.get('faces_detected', 0) > 0:
                print("✅ Visage détecté avec succès!")
            else:
                print("⚠️  Aucun visage détecté")
                
            if result.get('anomalies'):
                print(f"⚠️  Anomalies détectées: {len(result['anomalies'])}")
            else:
                print("✅ Aucune anomalie détectée")
                
        else:
            print(f"❌ Erreur: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur de connexion: {e}")
    except Exception as e:
        print(f"❌ Erreur: {e}")

def test_health():
    """Tester la santé du service"""
    print("🏥 Test de santé du service...")
    
    url = "https://python-proctoring.onrender.com/health"
    
    try:
        response = requests.get(url, timeout=10)
        print(f"📥 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Service Status: {result.get('status')}")
            print(f"✅ Active Sessions: {result.get('active_sessions')}")
        else:
            print(f"❌ Erreur: {response.text}")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    print("🌐 Test du service de proctoring Render")
    print("=" * 50)
    
    # Test 1: Santé du service
    test_health()
    print()
    
    # Test 2: Détection de visage
    test_face_detection()
    
    print("\n🎯 Tests terminés!")
