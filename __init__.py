# __init__.py principal del proyecto de Sistema Metacognitivo
# Facilita la importación de todos los componentes desde cualquier parte

import os
import sys

# Configurar todas las rutas importantes
project_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(project_dir, 'core')
training_dir = os.path.join(project_dir, 'training')
utils_dir = os.path.join(project_dir, 'utils')
tokenization_dir = os.path.join(project_dir, 'tokenization')

# Añadir todas las rutas al path de Python
for path in [project_dir, core_dir, training_dir, utils_dir, tokenization_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Importar submódulos para facilitar su acceso
try:
    import core
    import training
    import utils
    import tokenization
    
    # Exponer los componentes principales directamente
    from core import (
        SistemaMetacognitivoIntegrado, 
        CerebroAutonomo,
        AutoObservacion,
        ReflexionMetacognitiva,
        AutoModificacion
    )
    
    from training import (
        entrenar_modelo_metacognitivo_directo,
        EntrenadorMetacognitivoDirecto
    )
    
    from utils import (
        crear_modelo_y_tokenizador_nuevos,
        extraer_textos_de_materiales
    )
    
    __all__ = [
        'core',
        'training',
        'utils',
        'tokenization',
        'SistemaMetacognitivoIntegrado',
        'CerebroAutonomo',
        'AutoObservacion',
        'ReflexionMetacognitiva',
        'AutoModificacion',
        'entrenar_modelo_metacognitivo_directo',
        'EntrenadorMetacognitivoDirecto',
        'crear_modelo_y_tokenizador_nuevos',
        'extraer_textos_de_materiales'
    ]
    
except ImportError as e:
    print(f"Error al importar submódulos en el __init__.py principal: {e}")
