# Auditoría rápida de la presentación

Revisión del 21 de septiembre de 2026. Alcance: cotejo documental del paper corregido, reportes archivados, implementación recuperada y evidencia agregada pública; comprobación de cálculos y revisión de la presentación. No se reentrenaron modelos ni se realizó una nueva evaluación de las etiquetas de los textos sintéticos.

## Resultado del cotejo

Las cifras principales de las diapositivas coinciden con las fuentes revisadas. La revisión simplifica su exposición sin modificar resultados ni presentar como demostrada la transferencia a otras instituciones.

| Tema | Evidencia comprobada | Forma de presentarlo |
|---|---|---|
| Tres vías de modelado | Pilotos generativos con prompting/LoRA, TF-IDF con clasificadores clásicos y encoders de texto en español. | Tres estrategias; los LLM generativos también usan Transformers. |
| Comparación generativa | Mismos 133 reportes: exactitud conjunta 0.2331 para SGD, 0.1955 para Gemma LoRA y 0.1053 para Qwen prompting. Piloto LoRA: 625 ejemplos y 80 pasos. | Panel identificado como piloto, separado de los resultados sobre desarrollo completo. No demuestra una superioridad universal de una familia. |
| Referencias clásicas | Severidad: referencia temprana 0.4974 de macro-F1; BETO 0.5429. Tipos: SVM de caracteres 0.7642 de macro-F1; BETO con umbral fijo 0.45 alcanza 0.819845. | Se conserva la identificación de referencia clásica temprana; hubo ensambles clásicos posteriores más fuertes. |
| Aumentación de Severe | 616/13,630 pasa a 1,132/14,146 tras añadir 516 registros: 4.52% a 8.00%. | Balance parcial mediante composición de plantillas condicionada por indicios del texto; originales conservados y etiquetas heredadas. No se describe como paráfrasis validada ni como generación con LLM. |
| Resultado de la receta aumentada | Sobre 3,405 reportes originales: macro-F1 0.548114 a 0.555198; F1 de Severe 0.417808 a 0.433460; recall de Severe 0.396104 a 0.370130. | Diferencias absolutas multiplicadas por cien: +0.71, +1.57 y -2.60 puntos. Se mantiene visible la caída de recall. |
| Arquitectura de tipos | El código final de S2 usa seis salidas sigmoid y deriva N/A si ninguna alcanza 0.45. La cabeza auxiliar de la receta S1 usa siete objetivos suaves. | Los diagramas distinguen ambas cabezas; se sigue la implementación específica y no la descripción genérica de siete salidas del paper. |
| Consenso de anotación | 877/1,212 reportes de desarrollo, 72.36%, presentan votos divididos en al menos uno de los seis tipos. Proporciones de tres votos; varianza interna p(1-p). | Análisis descriptivo añadido después del paper, con denominador explícito. No se llama varianza entre trabajadores sociales identificados. |
| Ancla y ajuste final | 177 Medium a Mild, 101 Mild a Medium y 12 High a Severe suman 290 de 4,219. Se conservan 3,929 predicciones, 93.13%. | Ajuste restringido alrededor del ancla; no se presenta como una nueva arquitectura. |
| Resultado oficial | Severidad: 0.60798 de macro-F1, puesto 3/16. Tipos: 0.87016 de micro-F1, puesto 6/15. Severidad pasa de 0.60678 a 0.60798 y tipos permanece igual. | Resultados del benchmark. Se conserva una frase indicando que las puntuaciones oficiales orientaron la selección. |

Fuentes públicas: [resultados y protocolos](../docs/results.md), [método](../docs/method.md), [evidencia agregada](../results/evidence.json), [consenso de anotación](../docs/annotation_consensus.md) e [implementación](../scripts/).

El cotejo adicional empleó el paper corregido de junio, `model_matrix_balanced133_final_20260419.md`, `reporte_subtask2_encoder_20260429.md` y `augrep08_metrics_summary.json`. Los registros individuales y los archivos privados de experimentación no forman parte de esta publicación.

## Decisión editorial: qué queda en pantalla y qué pasa a las notas

En pantalla se conservan los datos necesarios para interpretar cada gráfico: partición, muestra, métrica, referencia temprana, balance parcial, caída del recall y uso de puntuaciones oficiales para la selección. El análisis de anotación sigue identificado como posterior al paper. La transferencia entre instituciones aparece como trabajo futuro.

Las notas y la sección de respaldo para preguntas de los guiones conservan los detalles que interrumpían la exposición: presupuestos de entrenamiento distintos, necesidad de aislar los efectos de estilo/balance/pesos, posible introducción de hechos por las plantillas, discrepancias entre etiquetas duras y suaves, y ausencia de identidades individuales y votos de severidad. Estos puntos no se consideran resueltos por una revisión documental; se trasladan al respaldo técnico.

La narrativa principal sigue el problema, las tres vías exploradas, las decisiones de arquitectura, el balance de Severe y el ajuste final del ancla. La presentación no usa la pequeña ganancia final como explicación principal del sistema.

## Material para el evento

La versión para enviar tiene diez diapositivas principales en inglés, dos diapositivas finales de respaldo para preguntas y un PDF visual de respaldo. Las versiones españolas y los guiones son para revisión y ensayo. El póster conserva el formato A0 vertical (841 x 1189 mm), las referencias enlazadas al repositorio público y el código QR.

El correo organizativo del 19 de septiembre pide enviar las diapositivas a más tardar el 21 de septiembre. La [guía oficial de montaje](https://sepln2026.org/guia-de-colocacion-de-posters/) pide llevar el póster impreso y no impone una plantilla. Esta carpeta no implica que el correo ya haya sido enviado.

## Ampliación del 22 de septiembre: ensamble con el modelo aumentado

La diapositiva 11 compara BETO base, BETO con aumentación Severe y el candidato intermedio `repro_diverse10_dw0.50_equal_ordmix` del 5 de mayo. Este incluye `augrep08` y su reporte documenta que la reconstrucción reprodujo el score de desarrollo. Sus resultados son macro-F1 0.580536, F1 Severe 0.473684 y recall Severe 0.409091. Se mantienen los diez minutos del discurso principal y se reserva la diapositiva adicional para preguntas.

La explicación previa citó 41.56% de recall de un candidato anterior del 3 de mayo sin recoger el problema de reconstrucción documentado al final del mismo reporte. Esa omisión queda corregida: la nueva diapositiva utiliza el 40.91% del candidato posterior con reconstrucción verificada en el reporte. Ninguno de esos valores debe identificarse como el recall del sistema final. La comparación corresponde al ensamble completo y no a una ablación del modelo aumentado. Véase [la evidencia y el historial de la corrección](../docs/augmentation_ensemble.md).


## Revisión histórica integral del 22 de septiembre

La ampliación anterior queda como registro del proceso. La diapositiva 11 actual compara el sistema realmente enviado: ancla Top10, foldbag BETO aumentado + Electricidad y router final sobre los mismos 3,405 reportes de desarrollo. Sus recalls Severe son 40.91%, 40.26% y 43.51%. El router conserva 63 aciertos y añade cuatro, junto con nueve falsos positivos adicionales. Los valores se recalcularon desde las predicciones guardadas y coinciden con el JSON del paquete.

La comparación directa de los ZIP confirma que el test final conserva las 116 predicciones Severe del ancla y añade doce. Por inclusión del conjunto predicho positivo, su recall no puede bajar respecto al ancla sobre el mismo gold y alineación; su valor absoluto sigue desconocido. Esta propiedad no garantiza precisión ni F1. El micro-F1 devel correcto del router es 0.58825257, frente a 0.588840 en un reporte narrativo anterior; no cambia el resultado oficial.

La diapositiva 7 ahora explica la mejora externa de S2 y el resultado negativo de highpen en S1. El análisis posterior de consenso pasa a la diapositiva 12 de respaldo. El ensamble intermedio permanece documentado, pero ya no cierra la explicación del recall final. Fuentes y valores en [la reconstrucción de la selección final](../docs/final_selection.md).
