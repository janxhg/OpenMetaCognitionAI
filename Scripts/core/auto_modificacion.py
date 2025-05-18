#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de Auto-Modificación para el Sistema Metacognitivo Avanzado.

Este módulo permite que el modelo modifique sus propios parámetros
basándose en las reflexiones metacognitivas, implementando un ciclo
completo de auto-mejora.
"""

import os
import sys
import logging
import torch
import numpy as np
import json
import re
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Importar módulos del sistema metacognitivo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auto_observacion import AutoObservacion
from reflexion_metacognitiva import ReflexionMetacognitiva

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AutoModificacion:
    """
    Clase que implementa la capacidad de auto-modificación para modelos de lenguaje.
    Permite que el modelo modifique sus propios parámetros basándose en reflexiones metacognitivas.
    """
    
    def __init__(self, modelo: PreTrainedModel, tokenizer: AutoTokenizer, 
                 reflexion: ReflexionMetacognitiva = None, config: Dict = None):
        """
        Inicializa el módulo de auto-modificación.
        
        Args:
            modelo: Modelo pre-entrenado a modificar
            tokenizer: Tokenizador asociado al modelo
            reflexion: Instancia opcional de ReflexionMetacognitiva
            config: Configuración del módulo de auto-modificación
        """
        self.modelo = modelo
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = next(modelo.parameters()).device
        
        # Crear módulo de reflexión si no se proporciona
        if reflexion is None:
            auto_obs = AutoObservacion(modelo)
            self.reflexion = ReflexionMetacognitiva(modelo, tokenizer, auto_obs)
        else:
            self.reflexion = reflexion
        
        # Configuración por defecto
        self.config.setdefault('max_modificacion_peso', 0.1)  # Máxima modificación permitida (10%)
        self.config.setdefault('umbral_confianza', 0.7)      # Umbral de confianza para aplicar modificaciones
        self.config.setdefault('modo_seguro', True)          # Modo seguro (limita modificaciones)
        self.config.setdefault('guardar_respaldo', True)     # Guardar respaldo antes de modificar
        
        # Historial de modificaciones
        self.historial_modificaciones = []
        
        # Respaldo del modelo original
        self.respaldo_modelo = None
        
        logger.info(f"Módulo de auto-modificación inicializado para modelo {type(modelo).__name__}")
    
    def _crear_respaldo(self):
        """Crea un respaldo de los parámetros actuales del modelo."""
        if self.config['guardar_respaldo']:
            self.respaldo_modelo = {
                nombre: param.clone().detach()
                for nombre, param in self.modelo.named_parameters()
                if param.requires_grad
            }
            logger.info("Respaldo de parámetros del modelo creado")
    
    def _restaurar_respaldo(self):
        """Restaura el modelo desde el respaldo."""
        if self.respaldo_modelo is not None:
            with torch.no_grad():
                for nombre, param in self.modelo.named_parameters():
                    if nombre in self.respaldo_modelo:
                        param.copy_(self.respaldo_modelo[nombre])
            logger.info("Modelo restaurado desde respaldo")
        else:
            logger.warning("No se puede restaurar: no hay respaldo disponible")
    
    def analizar_y_generar_plan(self, datos_adicionales: Dict = None) -> Dict:
        """
        Analiza el estado del modelo y genera un plan de modificación.
        
        Args:
            datos_adicionales: Datos adicionales para la reflexión
            
        Returns:
            Plan de modificación
        """
        # Generar reflexión metacognitiva
        reflexion_texto = self.reflexion.generar_reflexion(datos_adicionales)
        
        # Analizar problemas de aprendizaje
        problemas = self.reflexion.analizar_problemas_aprendizaje()
        
        # Generar estrategias de mejora
        estrategias = self.reflexion.generar_estrategias_mejora(problemas)
        
        # Extraer modificaciones sugeridas del texto de reflexión
        modificaciones_sugeridas = self._extraer_modificaciones_de_texto(reflexion_texto)
        
        # Combinar estrategias algorítmicas con sugerencias del texto
        plan = {
            'reflexion': reflexion_texto,
            'problemas': problemas,
            'estrategias': estrategias,
            'modificaciones_sugeridas': modificaciones_sugeridas,
            'plan_final': self._generar_plan_final(estrategias, modificaciones_sugeridas)
        }
        
        return plan
    
    def _extraer_modificaciones_de_texto(self, texto: str) -> List[Dict]:
        """
        Extrae sugerencias de modificación del texto de reflexión.
        
        Args:
            texto: Texto de reflexión metacognitiva
            
        Returns:
            Lista de modificaciones sugeridas
        """
        modificaciones = []
        
        # Buscar sección de modificaciones sugeridas
        match = re.search(r'MODIFICACIONES SUGERIDAS:(.*?)(?:$|^#)', texto, re.DOTALL | re.MULTILINE)
        if not match:
            return modificaciones
        
        seccion_modificaciones = match.group(1).strip()
        
        # Patrones para diferentes tipos de modificaciones
        patrones = [
            # Patrón para modificación de pesos específicos
            (r'(?:modificar|ajustar)(?:\s+los)?(?:\s+pesos)?(?:\s+de)?(?:\s+la)?(?:\s+capa)?\s+["\']?([\w\.\-]+)["\']?.*?(?:por|a|en)\s+([\d\.]+)', 
             'modificar_pesos'),
            
            # Patrón para reinicialización de capas
            (r'(?:reinicializar|resetear)(?:\s+los)?(?:\s+pesos)?(?:\s+de)?(?:\s+la)?(?:\s+capa)?\s+["\']?([\w\.\-]+)["\']?', 
             'reinicializar'),
            
            # Patrón para ajuste de learning rate
            (r'(?:aumentar|incrementar|reducir|disminuir)(?:\s+el)?\s+learning\s*rate.*?(?:por|a|en)\s+(?:un\s+)?(\d+(?:\.\d+)?)(?:\s*%|\s+veces)?', 
             'ajustar_lr')
        ]
        
        # Buscar coincidencias para cada patrón
        for linea in seccion_modificaciones.split('\n'):
            linea = linea.strip()
            if not linea:
                continue
                
            for patron, tipo in patrones:
                match = re.search(patron, linea, re.IGNORECASE)
                if match:
                    if tipo == 'modificar_pesos':
                        capa = match.group(1)
                        valor = float(match.group(2))
                        modificaciones.append({
                            'tipo': 'modificar_pesos',
                            'capa': capa,
                            'valor': valor,
                            'descripcion': linea
                        })
                    elif tipo == 'reinicializar':
                        capa = match.group(1)
                        modificaciones.append({
                            'tipo': 'reinicializar',
                            'capa': capa,
                            'descripcion': linea
                        })
                    elif tipo == 'ajustar_lr':
                        factor = float(match.group(1))
                        # Determinar si es aumento o disminución
                        if re.search(r'aumentar|incrementar', linea, re.IGNORECASE):
                            operacion = 'aumentar'
                        else:
                            operacion = 'disminuir'
                            
                        modificaciones.append({
                            'tipo': 'ajustar_lr',
                            'factor': factor,
                            'operacion': operacion,
                            'descripcion': linea
                        })
                    break
        
        return modificaciones
    
    def _generar_plan_final(self, estrategias: List[Dict], modificaciones_sugeridas: List[Dict]) -> List[Dict]:
        """
        Genera un plan final combinando estrategias algorítmicas y sugerencias textuales.
        
        Args:
            estrategias: Estrategias generadas algorítmicamente
            modificaciones_sugeridas: Modificaciones extraídas del texto
            
        Returns:
            Plan final de modificaciones
        """
        plan_final = []
        
        # Añadir estrategias algorítmicas
        for estrategia in estrategias:
            if estrategia['accion']['tipo'] == 'modificar_hiperparametro':
                plan_final.append({
                    'tipo': 'hiperparametro',
                    'parametro': estrategia['accion']['parametro'],
                    'valor_propuesto': estrategia['accion']['valor_propuesto'],
                    'confianza': 0.8,
                    'fuente': 'algoritmica',
                    'descripcion': estrategia['descripcion']
                })
            elif estrategia['accion']['tipo'] == 'reinicializar_pesos':
                for patron in estrategia['accion']['capas_objetivo']:
                    plan_final.append({
                        'tipo': 'reinicializar',
                        'patron_capa': patron,
                        'metodo': estrategia['accion']['metodo'],
                        'parametros': {'gain': estrategia['accion'].get('gain', 1.0)},
                        'confianza': 0.7,
                        'fuente': 'algoritmica',
                        'descripcion': estrategia['descripcion']
                    })
        
        # Añadir modificaciones sugeridas por el texto
        for mod in modificaciones_sugeridas:
            if mod['tipo'] == 'modificar_pesos':
                plan_final.append({
                    'tipo': 'modificar_pesos',
                    'capa': mod['capa'],
                    'valor': mod['valor'],
                    'confianza': 0.6,  # Menor confianza para sugerencias textuales
                    'fuente': 'reflexion',
                    'descripcion': mod['descripcion']
                })
            elif mod['tipo'] == 'reinicializar':
                plan_final.append({
                    'tipo': 'reinicializar',
                    'capa': mod['capa'],
                    'metodo': 'xavier_uniform',  # Método por defecto
                    'parametros': {'gain': 1.0},
                    'confianza': 0.6,
                    'fuente': 'reflexion',
                    'descripcion': mod['descripcion']
                })
            elif mod['tipo'] == 'ajustar_lr':
                plan_final.append({
                    'tipo': 'hiperparametro',
                    'parametro': 'learning_rate',
                    'operacion': mod['operacion'],
                    'factor': mod['factor'],
                    'confianza': 0.65,
                    'fuente': 'reflexion',
                    'descripcion': mod['descripcion']
                })
        
        # Filtrar por umbral de confianza
        plan_final = [
            accion for accion in plan_final 
            if accion['confianza'] >= self.config['umbral_confianza']
        ]
        
        return plan_final
    
    def aplicar_modificaciones(self, plan: Dict) -> Dict:
        """
        Aplica las modificaciones propuestas en el plan al modelo.
        
        Args:
            plan: Plan de modificación generado por analizar_y_generar_plan
            
        Returns:
            Resultados de las modificaciones aplicadas
        """
        # Crear respaldo antes de modificar
        self._crear_respaldo()
        
        resultados = {
            'modificaciones_aplicadas': [],
            'modificaciones_rechazadas': [],
            'errores': []
        }
        
        # Aplicar cada modificación del plan final
        for accion in plan['plan_final']:
            try:
                if accion['tipo'] == 'modificar_pesos':
                    exito = self._aplicar_modificacion_pesos(accion)
                elif accion['tipo'] == 'reinicializar':
                    exito = self._aplicar_reinicializacion(accion)
                elif accion['tipo'] == 'hiperparametro':
                    exito = self._aplicar_cambio_hiperparametro(accion)
                else:
                    exito = False
                    resultados['errores'].append({
                        'accion': accion,
                        'mensaje': f"Tipo de acción desconocido: {accion['tipo']}"
                    })
                
                if exito:
                    resultados['modificaciones_aplicadas'].append(accion)
                else:
                    resultados['modificaciones_rechazadas'].append(accion)
            
            except Exception as e:
                logger.error(f"Error al aplicar modificación: {str(e)}")
                resultados['errores'].append({
                    'accion': accion,
                    'mensaje': str(e)
                })
                resultados['modificaciones_rechazadas'].append(accion)
        
        # Registrar en historial
        self.historial_modificaciones.append({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'plan': plan,
            'resultados': resultados
        })
        
        return resultados
    
    def _aplicar_modificacion_pesos(self, accion: Dict) -> bool:
        """
        Aplica una modificación a los pesos de una capa específica.
        
        Args:
            accion: Especificación de la modificación a aplicar
            
        Returns:
            True si la modificación se aplicó correctamente, False en caso contrario
        """
        nombre_capa = accion['capa']
        valor = accion['valor']
        
        # Buscar parámetro por nombre exacto o por patrón
        parametro_encontrado = False
        with torch.no_grad():
            for nombre, param in self.modelo.named_parameters():
                if nombre == nombre_capa or (nombre_capa in nombre and '.' in nombre):
                    # Limitar la modificación según configuración de seguridad
                    if self.config['modo_seguro']:
                        valor_limitado = min(abs(valor), self.config['max_modificacion_peso'])
                        if valor < 0:
                            valor_limitado = -valor_limitado
                    else:
                        valor_limitado = valor
                    
                    # Aplicar modificación
                    if isinstance(valor_limitado, float):
                        # Si es un valor absoluto, establecer directamente
                        param.fill_(valor_limitado)
                    else:
                        # Si es un factor, multiplicar por el valor actual
                        param.mul_(valor_limitado)
                    
                    parametro_encontrado = True
                    logger.info(f"Modificados pesos de {nombre} con valor {valor_limitado}")
                    break
        
        return parametro_encontrado
    
    def _aplicar_reinicializacion(self, accion: Dict) -> bool:
        """
        Reinicializa los pesos de una capa específica.
        
        Args:
            accion: Especificación de la reinicialización a aplicar
            
        Returns:
            True si la reinicialización se aplicó correctamente, False en caso contrario
        """
        # Determinar el patrón de capa a reinicializar
        if 'capa' in accion:
            patron_capa = accion['capa']
            es_regex = False
        elif 'patron_capa' in accion:
            patron_capa = accion['patron_capa']
            es_regex = True
        else:
            return False
        
        # Determinar método de inicialización
        metodo = accion.get('metodo', 'xavier_uniform')
        parametros = accion.get('parametros', {})
        
        # Contar capas afectadas
        capas_afectadas = 0
        
        with torch.no_grad():
            for nombre, param in self.modelo.named_parameters():
                coincide = False
                if es_regex:
                    coincide = bool(re.match(patron_capa, nombre))
                else:
                    coincide = (nombre == patron_capa or patron_capa in nombre)
                
                if coincide and param.requires_grad:
                    # Aplicar método de inicialización
                    if metodo == 'xavier_uniform':
                        gain = parametros.get('gain', 1.0)
                        torch.nn.init.xavier_uniform_(param, gain=gain)
                    elif metodo == 'xavier_normal':
                        gain = parametros.get('gain', 1.0)
                        torch.nn.init.xavier_normal_(param, gain=gain)
                    elif metodo == 'kaiming_uniform':
                        torch.nn.init.kaiming_uniform_(param)
                    elif metodo == 'kaiming_normal':
                        torch.nn.init.kaiming_normal_(param)
                    elif metodo == 'zeros':
                        torch.nn.init.zeros_(param)
                    elif metodo == 'ones':
                        torch.nn.init.ones_(param)
                    else:
                        # Método desconocido, usar xavier_uniform por defecto
                        torch.nn.init.xavier_uniform_(param)
                    
                    capas_afectadas += 1
                    logger.info(f"Reinicializada capa {nombre} usando método {metodo}")
        
        return capas_afectadas > 0
    
    def _aplicar_cambio_hiperparametro(self, accion: Dict) -> bool:
        """
        Aplica un cambio a un hiperparámetro del entrenamiento.
        Nota: Esta función no modifica directamente el modelo sino que registra
        el cambio para ser aplicado en el optimizador o scheduler.
        
        Args:
            accion: Especificación del cambio de hiperparámetro
            
        Returns:
            True si el cambio se registró correctamente
        """
        # En esta implementación simplemente registramos el cambio
        # En un sistema real, esto modificaría el optimizador o scheduler
        parametro = accion['parametro']
        
        if 'valor_propuesto' in accion:
            valor = accion['valor_propuesto']
            logger.info(f"Registrado cambio de hiperparámetro {parametro} a {valor}")
        elif 'operacion' in accion and 'factor' in accion:
            operacion = accion['operacion']
            factor = accion['factor']
            logger.info(f"Registrado cambio de hiperparámetro {parametro}: {operacion} por factor {factor}")
        else:
            return False
        
        return True
    
    def ejecutar_ciclo_metacognitivo(self, datos_adicionales: Dict = None) -> Dict:
        """
        Ejecuta un ciclo completo de metacognición: observación, reflexión y modificación.
        
        Args:
            datos_adicionales: Datos adicionales para la reflexión
            
        Returns:
            Resultados del ciclo metacognitivo
        """
        # 1. Observación y reflexión
        plan = self.analizar_y_generar_plan(datos_adicionales)
        
        # 2. Aplicar modificaciones
        resultados = self.aplicar_modificaciones(plan)
        
        # 3. Evaluar resultados (en un sistema real, esto requeriría evaluación del modelo)
        # Por ahora, simplemente devolvemos los resultados de las modificaciones
        
        return {
            'plan': plan,
            'resultados_modificaciones': resultados,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def guardar_historial(self, ruta_directorio: str):
        """
        Guarda el historial de modificaciones en archivos para análisis posterior.
        
        Args:
            ruta_directorio: Directorio donde guardar el historial
        """
        os.makedirs(ruta_directorio, exist_ok=True)
        
        # Guardar historial en formato JSON
        with open(os.path.join(ruta_directorio, "historial_modificaciones.json"), "w", encoding="utf-8") as f:
            # Convertir datos para JSON
            historial_json = []
            for entrada in self.historial_modificaciones:
                # Crear copia para no modificar el original
                entrada_json = entrada.copy()
                
                # Eliminar elementos no serializables
                if 'plan' in entrada_json and 'reflexion' in entrada_json['plan']:
                    entrada_json['plan']['reflexion_length'] = len(entrada_json['plan']['reflexion'])
                    entrada_json['plan']['reflexion'] = entrada_json['plan']['reflexion'][:1000] + "..." \
                        if len(entrada_json['plan']['reflexion']) > 1000 else entrada_json['plan']['reflexion']
                
                historial_json.append(entrada_json)
            
            json.dump(historial_json, f, indent=2)
        
        # Guardar último ciclo como texto
        if self.historial_modificaciones:
            ultimo_ciclo = self.historial_modificaciones[-1]
            with open(os.path.join(ruta_directorio, "ultimo_ciclo.txt"), "w", encoding="utf-8") as f:
                f.write(f"Ciclo metacognitivo del {ultimo_ciclo['timestamp']}\n\n")
                f.write(f"REFLEXIÓN:\n{ultimo_ciclo['plan']['reflexion']}\n\n")
                f.write("MODIFICACIONES APLICADAS:\n")
                for mod in ultimo_ciclo['resultados']['modificaciones_aplicadas']:
                    f.write(f"- {mod['descripcion']}\n")
        
        logger.info(f"Historial de modificaciones guardado en {ruta_directorio}")


# Función auxiliar para crear una instancia del módulo
def crear_modulo_auto_modificacion(modelo, tokenizer, reflexion=None, config=None):
    """
    Crea y configura un módulo de auto-modificación para un modelo.
    
    Args:
        modelo: Modelo pre-entrenado a modificar
        tokenizer: Tokenizador asociado al modelo
        reflexion: Instancia opcional de ReflexionMetacognitiva
        config: Configuración opcional
        
    Returns:
        Instancia de AutoModificacion
    """
    return AutoModificacion(modelo, tokenizer, reflexion, config)


# No necesitamos una función especial para datetime ya que lo importamos directamente


if __name__ == "__main__":
    # Ejemplo de uso
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Cargar un modelo pequeño para demostración
    modelo_nombre = "gpt2"  # Usar un modelo pequeño para pruebas
    modelo = AutoModelForCausalLM.from_pretrained(modelo_nombre)
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    
    # Crear módulo de auto-modificación
    auto_mod = crear_modulo_auto_modificacion(modelo, tokenizer)
    
    # Ejecutar ciclo metacognitivo
    resultados = auto_mod.ejecutar_ciclo_metacognitivo()
    
    # Mostrar resultados
    print("\nModificaciones aplicadas:")
    for mod in resultados['resultados_modificaciones']['modificaciones_aplicadas']:
        print(f"- {mod['descripcion']}")
    
    # Guardar historial
    auto_mod.guardar_historial("./modificaciones_demo")
