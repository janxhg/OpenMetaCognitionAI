# __init__.py para la carpeta utils
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
    from .crear_modelo_desde_cero import (
        crear_modelo_y_tokenizador_nuevos,
        extraer_textos_de_materiales
    )
    
    __all__ = [
        'crear_modelo_y_tokenizador_nuevos',
        'extraer_textos_de_materiales'
    ]
except ImportError as e:
    print(f"Error al importar módulos en utils/__init__.py: {e}")
