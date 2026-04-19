# Generer le fichier apres installation des packages
pip freeze > requirements.txt

# Utilite : permet a quiconque de reproduire exactement
# le meme environnement avec :
pip install -r requirements.txt