# WomenHelp 2026: clasificación de reportes de violencia

Código de investigación de Marcel Herrera Rendón, Universidad de Sonora. Reúne modelos clásicos, encoders Transformer en español, aumentación dirigida a Severe y un router final conservador.

[English](README.md) · [Resultados](docs/results.md) · [Método](docs/method.md) · [Reproducción](docs/reproduction.md) · [Póster y charla](presentation/README.md)

## Qué contiene

Probamos TF-IDF con clasificadores, prompting y LoRA de LLM generativos, y encoders como BETO, Electricidad y BERTin. Los LLM también usan Transformers; la distinción es su forma de uso. El sistema final combina un ensamble y router para severidad con un BETO dedicado para tipos de violencia.

- Severidad: Mild, Medium, High o Severe.
- Tipos: económica, física, patrimonial, psicológica, sexual y vicaria. Pueden coexistir. El BETO final tiene seis salidas sigmoid y deriva N/A si ninguna alcanza 0.45.
- Aumentación: 516 textos de estilo institucional construidos con plantillas e indicios del origen. Severe pasó de 4.52% a 8.00% del entrenamiento. Son sintéticos con etiquetas heredadas, no paráfrasis verificadas.
- Ancla: el router conservó 93.13% de las predicciones de referencia y limitó los cambios a tres transiciones.

El resultado oficial fue **0.60798 macro-F1 en severidad (3.º/16)** y **0.87016 micro-F1 en tipos (6.º/15)**. La aumentación mejoró F1, pero redujo recall Severe. Los puntajes oficiales orientaron la selección y no hemos demostrado transferencia entre instituciones.

## Probar sin datos privados

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/demo_public.py
```

La demo usa marcadores artificiales y no necesita GPU ni descargar modelos. Para entrenar, consultar [reproduction.md](docs/reproduction.md) y obtener los datos directamente de la organización. El corpus, los identificadores, predicciones individuales, checkpoints, historiales privados y formularios administrativos no forman parte de esta entrega.

Los resultados agregados y las decisiones de exposición se documentan con su protocolo. El análisis de consenso entre anotadores es nuevo y posterior al paper; no permite comparar trabajadores individuales ni medir desacuerdo sobre severidad.

Esta versión es una selección documentada del código del proyecto, con rutas portables. No promete reproducir exactamente la entrega oficial sin los artefactos originales. No se ha elegido todavía una licencia de software: la disponibilidad pública no equivale a una autorización general de reutilización.
