# Sistema Metacognitivo para Modelos de IA

Este repositorio contiene la implementación de un sistema metacognitivo avanzado para modelos de lenguaje, permitiendo que el modelo observe, reflexione y modifique su propio proceso de aprendizaje.

## Descripción General

El Sistema Metacognitivo es una arquitectura de inteligencia artificial que implementa capacidades metacognitivas en modelos de lenguaje. Esto permite que el modelo:

1. **Observe** sus propios pesos, gradientes y comportamiento durante el entrenamiento
2. **Reflexione** sobre su proceso de aprendizaje y estado interno
3. **Modifique** sus propios parámetros basándose en sus reflexiones
4. **Aprenda** de manera autónoma a partir de materiales de estudio

Esta implementación se basa en las ideas del aprendizaje metacognitivo, donde un sistema no solo aprende de los datos, sino que también aprende a aprender, ajustando sus estrategias y comportamiento basándose en la introspección sobre su propio rendimiento.

## Estructura del Repositorio

### Core - Núcleo del Sistema
- `cerebro_autonomo.py` - Implementación del cerebro autónomo para procesamiento cognitivo
- `auto_observacion.py` - Componente para analizar estados internos del modelo
- `reflexion_metacognitiva.py` - Generación de reflexiones sobre el proceso de aprendizaje
- `auto_modificacion.py` - Capacidad para proponer y aplicar modificaciones
- `ciclo_metacognitivo.py` - Implementación del ciclo completo de metacognición
- `aprendizaje_autonomo.py` - Implementación de memoria y biblioteca de conocimiento
- `componentes_basicos.py` - Definiciones básicas y clases útiles
- `sistema_metacognitivo_integrado.py` - Sistema que integra todos los componentes

### Training - Entrenamiento
- `entrenamiento_metacognitivo.py` - Entrenamiento con capacidades metacognitivas
- `entrenamiento_metacognitivo_directo.py` - Implementación del entrenamiento directo
- `entrenar_con_metacognitivo_directo.py` - Script para ejecutar entrenamiento directo

### Tokenization - Tokenización
- `tokenizador_metacognitivo.py` - Implementación del tokenizador con capacidades metacognitivas
- `entrenar_tokenizador_espanol.py` - Utilidad para entrenar tokenizadores en español

### Utils - Utilidades
- `crear_modelo_desde_cero.py` - Utilidad para crear modelos nuevos
- `probar_modelo_integrado.py` - Herramienta para probar modelos entrenados

## Ciclo Metacognitivo

El sistema implementa un ciclo metacognitivo completo que consta de cuatro fases principales:

1. **Fase de Aprendizaje**: El modelo aprende de materiales de estudio mediante entrenamiento
2. **Fase Introspectiva**: El sistema analiza los pesos internos y patrones del modelo
3. **Fase Reflexiva**: Genera reflexiones sobre el aprendizaje y el estado interno
4. **Fase de Auto-modificación**: Aplica cambios basados en las reflexiones

## Requisitos

- Python 3.8 o superior
- PyTorch 1.9 o superior
- Transformers 4.15 o superior
- CUDA recomendado para entrenamiento

## Ejemplo de Uso

Para entrenar un modelo utilizando el sistema metacognitivo directo:

```bash
python training/entrenar_con_metacognitivo_directo.py \
    --modelo_base nuevo \
    --dir_trabajo ./datos_entrenamiento \
    --dataset_externo ./datos/textos \
    --nivel_inteligencia 10 \
    --ciclos 25 \
    --pasos 1000 \
    --usar_cerebro_autonomo \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --fp16
```

## Licencia

Este proyecto está licenciado bajo:

MIT License

Copyright (c) 2025 	NeuroForge Labs

