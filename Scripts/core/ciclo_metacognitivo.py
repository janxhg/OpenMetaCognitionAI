#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Implementación del Ciclo Metacognitivo Completo para el Sistema Metacognitivo Avanzado.

Este script integra los tres componentes principales (Auto-Observación, Reflexión Metacognitiva
y Auto-Modificación) para implementar un ciclo completo de metacognición en modelos de lenguaje.
"""

import os
import sys
import logging
import argparse
import torch
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Importar componentes del sistema metacognitivo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_observacion import AutoObservacion, crear_modulo_auto_observacion
from reflexion_metacognitiva import ReflexionMetacognitiva, crear_modulo_reflexion
from auto_modificacion import AutoModificacion, crear_modulo_auto_modificacion

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CicloMetacognitivo:
    """
    Clase que implementa un ciclo completo de metacognición para modelos de lenguaje.
    Integra los tres componentes principales: Auto-Observación, Reflexión Metacognitiva
    y Auto-Modificación.
    """
    
    def __init__(self, modelo_path: str, directorio_salida: str, config: Dict = None):
        """
        Inicializa el ciclo metacognitivo.
        
        Args:
            modelo_path: Ruta al modelo pre-entrenado o nombre del modelo en HuggingFace
            directorio_salida: Directorio donde guardar resultados y modelos modificados
            config: Configuración del ciclo metacognitivo
        """
        self.modelo_path = modelo_path
        self.directorio_salida = directorio_salida
        self.config = config or {}
        
        # Configuración por defecto
        self.config.setdefault('num_ciclos', 3)  # Número de ciclos metacognitivos a ejecutar
        self.config.setdefault('evaluar_despues_de_ciclo', True)  # Evaluar después de cada ciclo
        self.config.setdefault('guardar_despues_de_ciclo', True)  # Guardar después de cada ciclo
        self.config.setdefault('max_modificacion_peso', 0.05)  # Limitar modificaciones al 5%
        
        # Crear directorios de salida
        os.makedirs(directorio_salida, exist_ok=True)
        
        # Cargar modelo y tokenizer
        logger.info(f"Cargando modelo desde {modelo_path}")
        self.modelo = AutoModelForCausalLM.from_pretrained(modelo_path)
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_path)
        
        # Detectar dispositivo
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo.to(self.device)
        logger.info(f"Modelo cargado en dispositivo: {self.device}")
        
        # Inicializar componentes
        self.auto_observacion = crear_modulo_auto_observacion(self.modelo)
        self.reflexion = crear_modulo_reflexion(self.modelo, self.tokenizer, self.auto_observacion)
        self.auto_modificacion = crear_modulo_auto_modificacion(
            self.modelo, 
            self.tokenizer,
            self.reflexion,
            {'max_modificacion_peso': self.config['max_modificacion_peso']}
        )
        
        # Historial de ciclos
        self.historial_ciclos = []
        
        logger.info("Ciclo metacognitivo inicializado")
    
    def ejecutar_ciclo(self, datos_adicionales: Dict = None, ciclo_num: int = 1) -> Dict:
        """
        Ejecuta un ciclo completo de metacognición.
        
        Args:
            datos_adicionales: Datos adicionales para la reflexión
            ciclo_num: Número de ciclo actual
            
        Returns:
            Resultados del ciclo
        """
        logger.info(f"Iniciando ciclo metacognitivo #{ciclo_num}")
        
        # Directorio para este ciclo
        directorio_ciclo = os.path.join(self.directorio_salida, f"ciclo_{ciclo_num}")
        os.makedirs(directorio_ciclo, exist_ok=True)
        
        # 1. Auto-observación
        logger.info("Fase 1: Auto-observación")
        self.auto_observacion.observar_pesos()
        self.auto_observacion.guardar_observaciones(os.path.join(directorio_ciclo, "observaciones"))
        
        # 2. Reflexión metacognitiva
        logger.info("Fase 2: Reflexión metacognitiva")
        reflexion_texto = self.reflexion.generar_reflexion(datos_adicionales)
        self.reflexion.guardar_reflexiones(os.path.join(directorio_ciclo, "reflexiones"))
        
        # 3. Auto-modificación
        logger.info("Fase 3: Auto-modificación")
        resultados_modificacion = self.auto_modificacion.ejecutar_ciclo_metacognitivo(datos_adicionales)
        self.auto_modificacion.guardar_historial(os.path.join(directorio_ciclo, "modificaciones"))
        
        # 4. Evaluación (si está configurado)
        resultados_evaluacion = None
        if self.config['evaluar_despues_de_ciclo']:
            logger.info("Fase 4: Evaluación")
            resultados_evaluacion = self._evaluar_modelo()
        
        # 5. Guardar modelo modificado (si está configurado)
        ruta_modelo_guardado = None
        if self.config['guardar_despues_de_ciclo']:
            logger.info("Fase 5: Guardando modelo modificado")
            ruta_modelo_guardado = os.path.join(directorio_ciclo, "modelo")
            self._guardar_modelo(ruta_modelo_guardado)
        
        # Registrar resultados del ciclo
        resultados_ciclo = {
            'ciclo_num': ciclo_num,
            'observaciones': self.auto_observacion.observaciones,
            'reflexion': reflexion_texto,
            'modificaciones': resultados_modificacion,
            'evaluacion': resultados_evaluacion,
            'modelo_guardado': ruta_modelo_guardado
        }
        
        self.historial_ciclos.append(resultados_ciclo)
        
        # Guardar resumen del ciclo
        with open(os.path.join(directorio_ciclo, "resumen_ciclo.json"), "w", encoding="utf-8") as f:
            # Crear versión serializable
            resumen = {
                'ciclo_num': ciclo_num,
                'reflexion_length': len(reflexion_texto),
                'reflexion_extracto': reflexion_texto[:500] + "..." if len(reflexion_texto) > 500 else reflexion_texto,
                'num_modificaciones_aplicadas': len(resultados_modificacion['resultados_modificaciones']['modificaciones_aplicadas']),
                'num_modificaciones_rechazadas': len(resultados_modificacion['resultados_modificaciones']['modificaciones_rechazadas']),
                'evaluacion': resultados_evaluacion,
                'modelo_guardado': ruta_modelo_guardado
            }
            json.dump(resumen, f, indent=2)
        
        logger.info(f"Ciclo metacognitivo #{ciclo_num} completado")
        return resultados_ciclo
    
    def ejecutar_ciclos_completos(self, num_ciclos: int = None) -> List[Dict]:
        """
        Ejecuta múltiples ciclos metacognitivos, acumulando los cambios entre ciclos.
        
        Args:
            num_ciclos: Número de ciclos a ejecutar (usa config si es None)
            
        Returns:
            Lista con resultados de todos los ciclos
        """
        if num_ciclos is None:
            num_ciclos = self.config['num_ciclos']
        
        resultados = []
        modelo_actual_path = self.modelo_path  # Comenzamos con el modelo original
        
        for i in range(1, num_ciclos + 1):
            logger.info(f"Ciclo {i}/{num_ciclos} - Usando modelo: {modelo_actual_path}")
            
            # Añadir datos adicionales con resultados de ciclos anteriores
            datos_adicionales = None
            if i > 1:
                datos_adicionales = {
                    'ciclo_anterior': {
                        'num': i - 1,
                        'modificaciones_aplicadas': len(resultados[-1]['modificaciones']['resultados_modificaciones']['modificaciones_aplicadas']),
                        'evaluacion': resultados[-1]['evaluacion']
                    }
                }
            
            # Si no es el primer ciclo, necesitamos cargar el modelo del ciclo anterior
            if i > 1:
                # Crear una nueva instancia de CicloMetacognitivo con el modelo modificado
                ciclo_actual = CicloMetacognitivo(modelo_actual_path, self.directorio_salida, self.config)
                # Ejecutar ciclo con el modelo actualizado
                resultado_ciclo = ciclo_actual.ejecutar_ciclo(datos_adicionales, i)
            else:
                # Para el primer ciclo, usamos la instancia actual
                resultado_ciclo = self.ejecutar_ciclo(datos_adicionales, i)
            
            resultados.append(resultado_ciclo)
            
            # Actualizar la ruta al modelo para el siguiente ciclo
            if i < num_ciclos:  # No necesitamos actualizar después del último ciclo
                modelo_actual_path = os.path.join(self.directorio_salida, f"ciclo_{i}", "modelo")
                logger.info(f"Próximo ciclo usará el modelo: {modelo_actual_path}")
        
        # Guardar resumen general
        self._guardar_resumen_general()
        
        return resultados
    
    def _evaluar_modelo(self) -> Dict:
        """
        Evalúa el modelo después de las modificaciones.
        En una implementación real, esto evaluaría el rendimiento en tareas específicas.
        
        Returns:
            Resultados de la evaluación
        """
        # Esta es una implementación simplificada
        # En un sistema real, se evaluaría el modelo en tareas específicas
        
        # Simulamos una evaluación básica
        return {
            'perplexity': 10.5,  # Valor simulado
            'accuracy': 0.85,    # Valor simulado
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _guardar_modelo(self, ruta_directorio: str) -> str:
        """
        Guarda el modelo modificado.
        
        Args:
            ruta_directorio: Directorio donde guardar el modelo
            
        Returns:
            Ruta al modelo guardado
        """
        os.makedirs(ruta_directorio, exist_ok=True)
        self.modelo.save_pretrained(ruta_directorio)
        self.tokenizer.save_pretrained(ruta_directorio)
        logger.info(f"Modelo guardado en {ruta_directorio}")
        return ruta_directorio
    
    def _guardar_resumen_general(self):
        """Guarda un resumen general de todos los ciclos ejecutados."""
        resumen = {
            'modelo_base': self.modelo_path,
            'num_ciclos': len(self.historial_ciclos),
            'ciclos': []
        }
        
        for i, ciclo in enumerate(self.historial_ciclos):
            resumen_ciclo = {
                'ciclo_num': i + 1,
                'num_modificaciones_aplicadas': len(ciclo['modificaciones']['resultados_modificaciones']['modificaciones_aplicadas']),
                'num_modificaciones_rechazadas': len(ciclo['modificaciones']['resultados_modificaciones']['modificaciones_rechazadas']),
                'evaluacion': ciclo['evaluacion']
            }
            resumen['ciclos'].append(resumen_ciclo)
        
        with open(os.path.join(self.directorio_salida, "resumen_general.json"), "w", encoding="utf-8") as f:
            json.dump(resumen, f, indent=2)
        
        logger.info(f"Resumen general guardado en {self.directorio_salida}/resumen_general.json")


# No necesitamos una función especial para datetime ya que lo importamos directamente


def main():
    """Función principal para ejecutar el ciclo metacognitivo desde línea de comandos."""
    parser = argparse.ArgumentParser(description="Ejecutar ciclo metacognitivo completo")
    parser.add_argument("--modelo", type=str, default="gpt2", 
                        help="Ruta al modelo o nombre del modelo en HuggingFace")
    parser.add_argument("--salida", type=str, default="./resultados_metacognitivos", 
                        help="Directorio de salida para resultados y modelos")
    parser.add_argument("--ciclos", type=int, default=3, 
                        help="Número de ciclos metacognitivos a ejecutar")
    parser.add_argument("--max_modificacion", type=float, default=0.05, 
                        help="Máxima modificación permitida para pesos (0.05 = 5%)")
    parser.add_argument("--no_evaluar", action="store_true", 
                        help="No evaluar después de cada ciclo")
    parser.add_argument("--no_guardar", action="store_true", 
                        help="No guardar modelo después de cada ciclo")
    
    args = parser.parse_args()
    
    # Configurar ciclo metacognitivo
    config = {
        'num_ciclos': args.ciclos,
        'evaluar_despues_de_ciclo': not args.no_evaluar,
        'guardar_despues_de_ciclo': not args.no_guardar,
        'max_modificacion_peso': args.max_modificacion
    }
    
    try:
        # Crear y ejecutar ciclo metacognitivo
        ciclo = CicloMetacognitivo(args.modelo, args.salida, config)
        resultados = ciclo.ejecutar_ciclos_completos()
        
        # Mostrar resumen
        print("\n" + "="*50)
        print("CICLO METACOGNITIVO COMPLETADO")
        print("="*50)
        print(f"Modelo base: {args.modelo}")
        print(f"Ciclos ejecutados: {args.ciclos}")
        print(f"Resultados guardados en: {args.salida}")
        print("="*50)
        
        return 0
    
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
