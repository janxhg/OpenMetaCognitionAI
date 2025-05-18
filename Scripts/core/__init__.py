# __init__.py para la carpeta core
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
    from .auto_observacion import AutoObservacion
    from .reflexion_metacognitiva import ReflexionMetacognitiva
    from .auto_modificacion import AutoModificacion
    from .cerebro_autonomo import CerebroAutonomo
    from .sistema_metacognitivo_integrado import SistemaMetacognitivoIntegrado
    
    __all__ = [
        'AutoObservacion',
        'ReflexionMetacognitiva',
        'AutoModificacion',
        'CerebroAutonomo',
        'SistemaMetacognitivoIntegrado'
    ]
except ImportError as e:
    print(f"Error al importar módulos en core/__init__.py: {e}")
