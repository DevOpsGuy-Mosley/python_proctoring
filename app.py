"""
Point d'entrée WSGI pour Render
Ce fichier permet à Gunicorn de trouver l'application Flask
"""
from proctoring_service import app, socketio

if __name__ == "__main__":
    socketio.run(app, debug=False)
