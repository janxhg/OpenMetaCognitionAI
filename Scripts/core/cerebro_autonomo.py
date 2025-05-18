#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cerebro Autónomo para el sistema de aprendizaje.
Este módulo implementa las capacidades cognitivas del sistema:
- Comprensión de textos
- Extracción de conceptos
- Razonamiento sobre el conocimiento
- Reflexión sobre el aprendizaje
"""

import os
import json
import logging
import re
import torch
import random
import numpy as np
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CerebroAutonomo:
    """Clase que implementa las capacidades cognitivas del sistema de aprendizaje autónomo."""
    
    def __init__(self, ruta_modelo=None, dispositivo=None):
        """Inicializa el cerebro autónomo."""
        self.modelo = None
        self.tokenizer = None
        self.dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cargar modelo si se proporciona ruta
        if ruta_modelo:
            self.cargar_modelo(ruta_modelo)
    
    def cargar_modelo(self, ruta_modelo):
        """Carga un modelo de lenguaje pre-entrenado."""
        try:
            logger.info(f"Cargando modelo desde {ruta_modelo}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
            self.modelo = AutoModelForCausalLM.from_pretrained(ruta_modelo)
            
            # Mover modelo al dispositivo adecuado
            self.modelo.to(self.dispositivo)
            
            logger.info(f"Modelo cargado correctamente en dispositivo: {self.dispositivo}")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            return False
    
    def generar_texto(self, prompt, max_length=150, temperatura=0.7, top_p=0.9, num_return=1):
        """Genera texto a partir de un prompt dado."""
        if not self.modelo or not self.tokenizer:
            logger.error("No se ha cargado ningún modelo")
            return ["Error: No se ha cargado ningún modelo"]
        
        # Limitar la longitud del prompt para evitar errores
        if len(prompt.split()) > 100:
            prompt_tokens = prompt.split()
            prompt = " ".join(prompt_tokens[:100])
            logger.warning("Prompt truncado para evitar errores de contexto")
        
        # Limitar max_length para evitar errores CUDA
        max_length = min(max_length, 200)  # Asegurar que no exceda 200 tokens
        
        try:
            # Tokenizar prompt
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.dispositivo)
            
            # Verificar que el prompt no exceda el contexto del modelo
            if inputs.input_ids.shape[1] > 400:
                logger.warning("Prompt demasiado largo, truncando")
                inputs.input_ids = inputs.input_ids[:, :400]
                if hasattr(inputs, 'attention_mask'):
                    inputs.attention_mask = inputs.attention_mask[:, :400]
            
            # Generar texto con manejo de errores más robusto
            with torch.no_grad():  # Evitar cálculos de gradiente innecesarios
                outputs = self.modelo.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperatura,
                    top_p=top_p,
                    num_return_sequences=num_return,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,  # Evitar repeticiones
                    no_repeat_ngram_size=3   # Evitar repetir n-gramas
                )
            
            # Decodificar salidas
            textos_generados = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            
            # Eliminar el prompt del texto generado y limpiar
            textos_limpios = []
            for texto in textos_generados:
                if texto.startswith(prompt):
                    texto_limpio = texto[len(prompt):].strip()
                else:
                    texto_limpio = texto.strip()
                
                # Limpiar texto (eliminar repeticiones extrañas)
                texto_limpio = re.sub(r'(\b\w+\b)\s+\1{2,}', r'\1', texto_limpio)
                textos_limpios.append(texto_limpio)
            
            return textos_limpios
        
        except Exception as e:
            logger.error(f"Error al generar texto: {str(e)}")
            return ["Ocurrió un error durante la generación de texto. Intentando con un prompt más simple."]
    
    def comprender_texto(self, texto):
        """Comprende un texto y extrae su significado principal."""
        prompt = f"Lee el siguiente texto y resume su significado principal en un párrafo:\n\n{texto}\n\nSignificado principal:"
        
        respuesta = self.generar_texto(prompt, max_length=150)[0]
        return respuesta
    
    def extraer_conceptos(self, texto, dominio):
        """Extrae conceptos clave de un texto para un dominio específico."""
        # Limitar la longitud del texto para evitar errores
        if len(texto) > 500:
            texto = texto[:500] + "..."
            logger.warning("Texto truncado para extracción de conceptos")
        
        # Prompt más estructurado y explícito
        prompt = f"""Analiza el siguiente texto del dominio de {dominio} y extrae exactamente 3 conceptos clave.

Texto: "{texto}"

Para cada concepto, sigue EXACTAMENTE este formato:
1. [NOMBRE DEL CONCEPTO]: [DEFINICIÓN BREVE]

Conceptos:"""
        
        # Intentar extraer conceptos con un prompt más estructurado
        respuesta = self.generar_texto(prompt, max_length=200, temperatura=0.5)[0]
        
        # Procesar la respuesta para extraer conceptos estructurados
        conceptos = {}
        
        # Patrones para extraer conceptos y definiciones
        patrones = [
            # Patrón 1: Formato numerado (1. Concepto: Definición)
            r"(\d+)\s*\.\s*([^:]+):\s*([^\n]+)",
            # Patrón 2: Formato con guiones (- Concepto: Definición)
            r"\-\s*([^:]+):\s*([^\n]+)",
            # Patrón 3: Formato simple (Concepto: Definición)
            r"([^\d\-\s][^:]+):\s*([^\n]+)"
        ]
        
        # Probar cada patrón
        for patron in patrones:
            try:
                if patron.startswith(r"\d"):
                    # Para el patrón numerado
                    coincidencias = re.findall(patron, respuesta)
                    for match in coincidencias:
                        if len(match) == 3:  # Si tiene 3 elementos (número, nombre, definición)
                            _, nombre, definicion = match
                        elif len(match) == 2:  # Si tiene 2 elementos (nombre, definición)
                            nombre, definicion = match
                        else:
                            continue  # Si tiene otra estructura, saltamos
                        
                        nombre = nombre.strip()
                        definicion = definicion.strip()
                        if nombre and definicion:
                            conceptos[nombre] = {
                                "definicion": definicion,
                                "dominio": dominio
                            }
                else:
                    # Para los otros patrones
                    coincidencias = re.findall(patron, respuesta)
                    for match in coincidencias:
                        if isinstance(match, tuple):
                            if len(match) == 2:  # Si tiene 2 elementos (nombre, definición)
                                nombre, definicion = match
                            else:  # Si tiene otra estructura, tomamos los dos primeros
                                nombre = match[0]
                                definicion = match[1] if len(match) > 1 else ""
                        else:  # Si no es una tupla, seguimos
                            continue
                        
                        nombre = nombre.strip()
                        definicion = definicion.strip()
                        if nombre and definicion:
                            conceptos[nombre] = {
                                "definicion": definicion,
                                "dominio": dominio
                            }
            except Exception as e:
                logger.warning(f"Error al procesar patrón: {str(e)}")
                continue  # Continuar con el siguiente patrón
        
        # Si no se encontraron conceptos, intentar con un enfoque más simple
        if not conceptos:
            # Buscar líneas que parezcan definiciones
            lineas = respuesta.split('\n')
            for linea in lineas:
                if ':' in linea:
                    partes = linea.split(':', 1)
                    nombre = partes[0].strip()
                    definicion = partes[1].strip()
                    
                    # Limpiar el nombre de numeraciones o símbolos
                    nombre = re.sub(r'^\d+\.\s*|^\-\s*|^\*\s*', '', nombre)
                    
                    if nombre and definicion:
                        conceptos[nombre] = {
                            "definicion": definicion,
                            "dominio": dominio
                        }
        
        # Si aún no hay conceptos, crear algunos genéricos basados en el dominio
        if not conceptos and texto:
            # Extraer palabras clave del texto
            palabras = re.findall(r'\b[A-Z][a-z]{3,}\b', texto)
            palabras_unicas = list(set(palabras))[:3]  # Tomar hasta 3 palabras que empiecen con mayúscula
            
            for i, palabra in enumerate(palabras_unicas):
                conceptos[palabra] = {
                    "definicion": f"Concepto importante en el dominio de {dominio}",
                    "dominio": dominio
                }
        
        logger.info(f"Extraídos {len(conceptos)} conceptos del texto")
        return conceptos
    
    def identificar_relaciones(self, concepto1, concepto2):
        """Identifica posibles relaciones entre dos conceptos."""
        prompt = f"Identifica la relación más importante entre estos dos conceptos:\n\nConcepto 1: {concepto1}\nConcepto 2: {concepto2}\n\nLa relación entre estos conceptos es:"
        
        respuesta = self.generar_texto(prompt, max_length=100)[0]
        
        # Simplificar la respuesta para obtener solo el tipo de relación
        tipos_relacion = [
            "causa-efecto", "parte-todo", "generalización-especialización", 
            "analogía", "contraste", "secuencia", "dependencia", "similitud"
        ]
        
        tipo_detectado = "asociación"  # Tipo por defecto
        
        for tipo in tipos_relacion:
            if tipo.lower() in respuesta.lower():
                tipo_detectado = tipo
                break
        
        return {
            "tipo": tipo_detectado,
            "descripcion": respuesta
        }
    
    def reflexionar_sobre_aprendizaje(self, conceptos_aprendidos, tiempo_estudio):
        """Genera una reflexión sobre el proceso de aprendizaje."""
        # Crear un resumen de los conceptos aprendidos
        resumen_conceptos = ", ".join(list(conceptos_aprendidos.keys())[:5])
        if len(conceptos_aprendidos) > 5:
            resumen_conceptos += f" y {len(conceptos_aprendidos) - 5} más"
        
        prompt = f"""
Reflexiona sobre tu proceso de aprendizaje después de estudiar durante {tiempo_estudio} minutos.
Has aprendido sobre: {resumen_conceptos}.

Considera:
1. ¿Qué conceptos fueron más difíciles de entender y por qué?
2. ¿Qué conexiones identificaste entre estos conceptos y tu conocimiento previo?
3. ¿Qué estrategias de aprendizaje fueron más efectivas?
4. ¿Cómo podrías mejorar tu comprensión de estos temas?

Reflexión sobre mi aprendizaje:
"""
        
        reflexion = self.generar_texto(prompt, max_length=300, temperatura=0.8)[0]
        return reflexion
    
    def evaluar_comprension(self, concepto, definicion):
        """Evalúa la comprensión de un concepto específico."""
        prompt = f"""
Evalúa mi comprensión del siguiente concepto:

Concepto: {concepto}
Mi definición: {definicion}

Evalúa en una escala del 1 al 10 qué tan bien he entendido este concepto.
Proporciona retroalimentación específica sobre qué aspectos he entendido correctamente y cuáles necesito mejorar.

Evaluación (1-10) y retroalimentación:
"""
        
        evaluacion = self.generar_texto(prompt, max_length=200)[0]
        
        # Extraer puntuación numérica
        patron_puntuacion = r"(\d+)\/10|(\d+)\s*\/\s*10|(\d+) de 10|(\d+)"
        coincidencias = re.search(patron_puntuacion, evaluacion)
        
        puntuacion = 5  # Valor por defecto
        if coincidencias:
            # Tomar el primer grupo no nulo
            for grupo in coincidencias.groups():
                if grupo:
                    puntuacion = int(grupo)
                    break
            
            # Asegurar que está en el rango 1-10
            puntuacion = max(1, min(10, puntuacion))
        
        return {
            "puntuacion": puntuacion,
            "retroalimentacion": evaluacion
        }
    
    def generar_preguntas(self, texto, num_preguntas=3):
        """Genera preguntas de comprensión sobre un texto."""
        prompt = f"""
Lee el siguiente texto y genera {num_preguntas} preguntas que evalúen la comprensión profunda del material:

{texto}

Preguntas:
"""
        
        preguntas_texto = self.generar_texto(prompt, max_length=200)[0]
        
        # Extraer preguntas individuales
        preguntas = []
        lineas = preguntas_texto.split('\n')
        
        for linea in lineas:
            # Buscar líneas que parezcan preguntas
            linea = linea.strip()
            if linea and (linea.endswith('?') or re.match(r'^\d+\.', linea)):
                # Limpiar numeración
                pregunta = re.sub(r'^\d+[\.\)]\s*', '', linea)
                preguntas.append(pregunta)
        
        return preguntas[:num_preguntas]  # Limitar al número solicitado
    
    def responder_pregunta(self, pregunta, contexto=None):
        """Responde una pregunta basándose en un contexto opcional."""
        if contexto:
            prompt = f"""
Basándote en el siguiente contexto, responde la pregunta de manera precisa y concisa:

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:
"""
        else:
            prompt = f"Responde la siguiente pregunta de manera precisa y concisa:\n\nPregunta: {pregunta}\n\nRespuesta:"
        
        respuesta = self.generar_texto(prompt, max_length=150)[0]
        return respuesta
    
    def planificar_estudio(self, objetivos, tiempo_disponible, conocimiento_previo=None):
        """Genera un plan de estudio basado en objetivos y restricciones."""
        if conocimiento_previo:
            resumen_conocimiento = ", ".join(list(conocimiento_previo.keys())[:5])
            if len(conocimiento_previo) > 5:
                resumen_conocimiento += f" y {len(conocimiento_previo) - 5} conceptos más"
        else:
            resumen_conocimiento = "ninguno"
        
        prompt = f"""
Crea un plan de estudio detallado para alcanzar los siguientes objetivos de aprendizaje:
{objetivos}

Tiempo disponible: {tiempo_disponible} minutos
Conocimiento previo: {resumen_conocimiento}

El plan debe incluir:
1. División del tiempo en sesiones
2. Temas a cubrir en cada sesión
3. Actividades específicas para cada tema
4. Estrategias de aprendizaje recomendadas
5. Métodos para evaluar el progreso

Plan de estudio:
"""
        
        plan = self.generar_texto(prompt, max_length=400, temperatura=0.7)[0]
        return plan
    
    def autoevaluar_conocimiento(self, dominio):
        """Realiza una autoevaluación del conocimiento en un dominio específico."""
        prompt = f"""
Realiza una autoevaluación honesta de tu conocimiento actual sobre {dominio}.

Considera:
1. ¿Qué conceptos fundamentales dominas?
2. ¿Qué áreas específicas necesitan más estudio?
3. ¿Cómo calificarías tu nivel general de comprensión (principiante, intermedio, avanzado)?
4. ¿Qué conceptos erróneos o lagunas has identificado?

Autoevaluación de mi conocimiento en {dominio}:
"""
        
        autoevaluacion = self.generar_texto(prompt, max_length=300, temperatura=0.8)[0]
        
        # Extraer nivel de conocimiento
        nivel = "principiante"  # Valor por defecto
        if "avanzado" in autoevaluacion.lower():
            nivel = "avanzado"
        elif "intermedio" in autoevaluacion.lower():
            nivel = "intermedio"
        
        return {
            "texto": autoevaluacion,
            "nivel": nivel
        }
    
    def sintetizar_conocimiento(self, conceptos):
        """Sintetiza el conocimiento actual en una estructura coherente."""
        # Crear un resumen de los conceptos
        nombres_conceptos = list(conceptos.keys())
        
        if not nombres_conceptos:
            return "No hay suficientes conceptos para sintetizar."
        
        # Seleccionar algunos conceptos si hay demasiados
        if len(nombres_conceptos) > 10:
            nombres_seleccionados = random.sample(nombres_conceptos, 10)
        else:
            nombres_seleccionados = nombres_conceptos
        
        # Crear texto con los conceptos seleccionados
        texto_conceptos = ""
        for nombre in nombres_seleccionados:
            texto_conceptos += f"- {nombre}: {conceptos[nombre]['definicion']}\n"
        
        prompt = f"""
Sintetiza los siguientes conceptos en un texto coherente que muestre las relaciones entre ellos:

{texto_conceptos}

Síntesis de conocimiento:
"""
        
        sintesis = self.generar_texto(prompt, max_length=400, temperatura=0.7)[0]
        return sintesis
