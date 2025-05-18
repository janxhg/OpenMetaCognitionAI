# __init__.py para la carpeta training
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
    from .entrenamiento_metacognitivo import entrenar_modelo, TextoDataset, IntrospectionCallback
    from .entrenamiento_metacognitivo_directo import (
        entrenar_modelo_metacognitivo_directo,
        EntrenadorMetacognitivoDirecto,
        MetacognitiveHook,
        EntrenamientoDirectoCallback
    )
    
    __all__ = [
        'entrenar_modelo',
        'TextoDataset',
        'IntrospectionCallback',
        'entrenar_modelo_metacognitivo_directo',
        'EntrenadorMetacognitivoDirecto',
        'MetacognitiveHook',
        'EntrenamientoDirectoCallback'
    ]
except ImportError as e:
    print(f"Error al importar módulos en training/__init__.py: {e}")
