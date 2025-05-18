# __init__.py para la carpeta tokenization
# Este archivo facilita las importaciones entre módulos

import os
import sys

# Añadir las rutas necesarias al path de Python
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Directorio raíz del proyecto
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Re-exportar componentes importantes para facilitar su importación
try:
    from .tokenizador_metacognitivo import MetacognitivoTokenizer
    
    __all__ = [
        'MetacognitivoTokenizer'
    ]
except ImportError as e:
    print(f"Error al importar módulos en tokenization/__init__.py: {e}")
