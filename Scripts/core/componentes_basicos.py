#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo con versiones básicas/dummy de componentes introspectivos para usar
cuando hay problemas con la inicialización de los componentes originales.
"""

import logging

logger = logging.getLogger(__name__)

class ComponenteBasico:
    """Componente básico que implementa interfaces mínimas para todos los componentes introspectivos"""
    
    def __init__(self, modelo=None, tokenizer=None, *args, **kwargs):
        self.modelo = modelo
    
    def obtener_estadisticas_pesos(self, *args, **kwargs):
        logger.info("Usando versión básica de obtener_estadisticas_pesos")
        return {"mensaje": "Versión básica", "capas": []}
    
    def obtener_estadisticas_gradientes(self, *args, **kwargs):
        logger.info("Usando versión básica de obtener_estadisticas_gradientes")
        return {"mensaje": "Versión básica", "capas": []}
    
    def generar_reflexion(self, *args, **kwargs):
        logger.info("Usando versión básica de generar_reflexion")
        return "Reflexión básica generada por componente simplificado."
    
    def analizar_problemas_aprendizaje(self, *args, **kwargs):
        logger.info("Usando versión básica de analizar_problemas_aprendizaje")
        return {}
    
    def generar_estrategias_mejora(self, *args, **kwargs):
        logger.info("Usando versión básica de generar_estrategias_mejora")
        return []
    
    def ejecutar_ciclo_metacognitivo(self, *args, **kwargs):
        logger.info("Usando versión básica de ejecutar_ciclo_metacognitivo")
        return {
            "mensaje": "Ciclo básico completado", 
            "resultados_modificaciones": {
                "modificaciones_aplicadas": [], 
                "modificaciones_rechazadas": []
            }
        }
    
    def observar_pesos(self, *args, **kwargs):
        logger.info("Usando versión básica de observar_pesos")
        return {}

class AutoObservacionBasica(ComponenteBasico):
    """Versión básica de AutoObservacion"""
    pass

class ReflexionBasica(ComponenteBasico):
    """Versión básica de ReflexionMetacognitiva"""
    pass

class AutoModificacionBasica(ComponenteBasico):
    """Versión básica de AutoModificacion"""
    pass
