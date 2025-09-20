# 🚀 Guide de Déploiement Render

## 📋 Prérequis
- Compte GitHub avec le dépôt `python_proctoring`
- Compte Render (gratuit)
- Base de données TPE Cloud configurée

## 🌐 Déploiement sur Render

### 1. Créer un compte Render
1. Aller sur [render.com](https://render.com)
2. Se connecter avec votre compte GitHub
3. Autoriser l'accès au dépôt `python_proctoring`

### 2. Créer un nouveau service
1. Cliquer sur **"New +"** → **"Web Service"**
2. Connecter votre dépôt GitHub
3. Sélectionner le dépôt `DevOpsGuy-Mosley/python_proctoring`

### 3. Configuration du service
```
Name: select-proctoring
Environment: Python 3
Region: Oregon (US West)
Branch: master
Root Directory: (laisser vide)
Build Command: pip install -r requirements.txt
Start Command: python proctoring_service.py
```

### 4. Variables d'environnement
Ajouter ces variables dans Render :
```
DB_HOST = your-tpecloud-db-host.tpecloud.ci
DB_NAME = pgroupeb_select_recruitment
DB_USER = pgroupeb_guy
DB_PASS = OiQE02f[g5C6T]-z
PYTHON_VERSION = 3.12.0
```

### 5. Déploiement
1. Cliquer sur **"Create Web Service"**
2. Attendre le déploiement (2-3 minutes)
3. Récupérer l'URL du service (ex: `https://select-proctoring.onrender.com`)

## 🔗 Configuration TPE Cloud

### Modifier le fichier PHP
Dans `select/php_backend/config/database.php`, ajouter :
```php
// Service de proctoring
const PROCTORING_SERVICE_URL = 'https://select-proctoring.onrender.com';
```

### Tester la connexion
```bash
curl https://select-proctoring.onrender.com/health
```

## 📊 Monitoring

### Logs Render
- Accéder aux logs via le dashboard Render
- Surveiller les erreurs de connexion DB
- Vérifier les performances

### Santé du service
- Endpoint `/health` pour vérifier l'état
- Endpoint `/sessions` pour voir les sessions actives

## 🔧 Dépannage

### Erreurs communes
1. **Connexion DB échouée** : Vérifier les variables d'environnement
2. **Timeout** : Render gratuit a des limites de performance
3. **Dépendances** : Vérifier `requirements.txt`

### Solutions
1. **Upgrade Render** : Plan payant pour de meilleures performances
2. **Optimiser le code** : Réduire la charge CPU
3. **Cache** : Implémenter un système de cache

## 💰 Coûts
- **Render Free** : Gratuit avec limitations
- **Render Paid** : À partir de $7/mois pour de meilleures performances

## 🎯 Résultat
Votre service de proctoring sera accessible via :
`https://select-proctoring.onrender.com`

Et connecté à votre application PHP sur TPE Cloud !
