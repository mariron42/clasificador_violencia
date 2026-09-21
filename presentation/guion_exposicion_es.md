# Guion de exposición

10 diapositivas, 10 minutos. Los tiempos incluyen señalar los diagramas y hacer pausas. Ensayar antes del evento.

## 1. Clasificación de reportes de violencia (0:00–0:30)

Buenos días. Soy Marcel Herrera Rendón, de la Universidad de Sonora. Voy a contar la estrategia que seguimos para clasificar reportes de violencia en WomenHelp. El punto central es cómo el problema orientó la elección de modelos, las salidas y los datos de entrenamiento. Después de encontrar recetas útiles, limitamos los cambios finales alrededor de una referencia. El corpus ya se explica en otra sesión, por lo que me concentraré en la implementación.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md (corrected system-description paper).

## 2. Tres familias, tres formas de clasificar (0:30–1:30)

Exploramos tres familias. Los modelos clásicos convierten palabras y caracteres en vectores TF-IDF y después clasifican. Los LLM generativos reciben instrucciones y producen una respuesta estructurada; probamos prompting y adaptación LoRA. Los encoders Transformer, como BETO, aprenden una representación contextual y alimentan cabezas de clasificación. Los LLM también usan Transformers: la distinción aquí es su forma de uso, generativa frente a discriminativa. Los clásicos dieron referencias fuertes y diversidad; los encoders permitieron controlar las salidas. La vía generativa quedó fuera del sistema final. La elección respondió a nuestros experimentos y recursos, no a una superioridad universal de una familia.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md, model families and implementation.

## 3. La evidencia orientó la elección (1:30–2:20)

A la izquierda comparo exactamente los mismos 133 reportes. La exactitud conjunta exige acertar severidad y todos los tipos: TF-IDF con SGD obtuvo 0.2331, el piloto Gemma con LoRA 0.1955 y Qwen con prompting 0.1053. El piloto LoRA usó 625 ejemplos y 80 pasos. No representa el potencial de todos los LLM. A la derecha comparo recetas clásicas y BETO en desarrollo completo. En severidad, la referencia clásica temprana dio 0.4974 y BETO focal 0.5429 de macro-F1. Hubo ensambles clásicos posteriores de 0.5047 y 0.5215. En tipos, el SVM de caracteres dio 0.7642 de macro-F1 y 0.8449 de micro-F1; BETO con umbral fijo 0.45 dio 0.8198 y 0.8685. Son recetas distintas, no una ablación aislada de arquitectura. No mezclo ambas escalas porque cambian los datos y la métrica. La conclusión práctica fue conservar referencias clásicas y profundizar en encoders especializados.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md, classical/encoder and common 133-record comparisons.

## 4. El problema determina las salidas (2:20–3:35)

Severidad requiere una sola categoría ordenada: Mild, Medium, High o Severe. Los tipos pueden coexistir: económica, física, patrimonial, psicológica, sexual y vicaria. El diagrama superior muestra la receta BETO aumentada: normalizamos espacios, tokenizamos y truncamos a 512 tokens. La cabeza principal usa softmax, focal ponderada y una penalización ordinal de 0.05. Una cabeza auxiliar aprende siete etiquetas de tipos, incluida N/A, con etiquetas suaves y peso 0.3, solo donde existen. Esta máscara evita interpretar información ausente como negativa. El componente final de tipos es otro BETO: procesa hasta 384 tokens, tiene seis sigmoid y umbral 0.45, y deriva N/A cuando ninguna alcanza el umbral. El código confirma seis salidas, aunque una descripción general del paper habla de siete. Estas decisiones adaptan arquitecturas existentes al problema; no proponemos una arquitectura nueva.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/run_competitive_encoder_severity.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/run_subtask2_encoder_20260429.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md.

## 5. Aumentación dirigida a Severe (3:35–5:05)

Severe tenía 616 de 13,630 ejemplos, apenas 4.52%. Elegimos aumentar solo esa clase en entrenamiento. El método destacado no utiliza un LLM: combina plantillas de apertura, acciones, contexto y cierre con estilo de reporte institucional. Los indicios del original, como relación, horario o referencias a amenazas, condicionan parte de las elecciones. La semilla es estable por registro y generamos como máximo dos variantes por origen. Retenemos los originales y heredamos sus etiquetas. Añadimos 516 textos: ahora Severe tiene 1,132 de 14,146, es decir, 8%. Reducimos el desequilibrio, sin igualar las cuatro clases. Las plantillas pueden introducir hechos, por lo que estos textos no deben presentarse como paráfrasis verificadas. El proyecto también exploró sinónimos y variantes de Mild; la gráfica siguiente corresponde específicamente a las plantillas Severe.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/src/womenhelp_competition/text_augmentation.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/evidence.json.

## 6. La mejora incluye un intercambio de errores (5:05–6:15)

Para que se vea el efecto, la gráfica muestra cambios absolutos multiplicados por cien y conserva el cero: 0.71 puntos de macro-F1, 1.57 de F1 Severe y menos 2.60 de recall Severe. Al lado están los valores originales en escala cero a uno. El conjunto de desarrollo contiene los mismos 3,405 reportes, sin aumentación. La mejora en F1 no implica recuperar más casos Severe: el recall bajó. Además, al aumentar datos cambiaron las proporciones y los pesos de clase recalculados. No podemos adjudicar el resultado exclusivamente al estilo de las plantillas. Faltan un control de sobremuestreo equivalente, pesos controlados, varias semillas y una auditoría de etiquetas. Esta es evidencia de una receta concreta, no una garantía de mejora general.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/evidence.json; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md. Delta = 100*(augmented - baseline).

## 7. El desacuerdo permanece en las etiquetas (6:15–7:15)

Añadí este análisis después del envío del paper. Las etiquetas suaves de tipos toman valores cero, un tercio, dos tercios y uno: conservan la fracción de tres votos positivos. En desarrollo, 877 de 1,212 reportes, 72.36%, tienen al menos uno de los seis tipos con voto dividido. Psicológica registra 33% de decisiones divididas y sexual 6.44%, siempre sobre todos los reportes. Las prevalencias influyen, así que esta comparación no mide directamente la dificultad de cada categoría. La varianza dentro de una decisión binaria es p por uno menos p: cero en unanimidad y dos novenos en división. No tenemos identidades para comparar trabajadores, ni votos de severidad, y la fuente consultada solo identifica anotadores humanos. Encontramos además 24 discrepancias entre mayoría suave y etiqueta dura en desarrollo, cuya causa no conocemos. Esta evidencia invita a evaluar por consenso. La cabeza auxiliar S1 sí aprovechó etiquetas suaves, pero el componente S2 final usó duras; los experimentos S2 con suaves fueron posteriores al envío.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/annotation_consensus.md; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/annotation_consensus.json; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/analyze_annotations.py. New analysis after the paper.

## 8. El ancla limita los cambios finales (7:15–8:30)

Al final ya teníamos recetas útiles. En lugar de volver a entrenarlas para cada ajuste, fijamos como ancla las predicciones de la mejor entrega previa. Esa referencia procedía de un ensamble Top10 con tres BETO, tres Electricidad, tres componentes clásicos y un BERTin. Un candidato de cinco folds BETO aumentado y cinco Electricidad propone cambios mediante umbrales ordinales. El router solo admite Medium a Mild, Mild a Medium y High a Severe. Este último necesita además indicios léxicos severos. Si la propuesta no cumple la regla, se conserva el ancla. Cambiaron 290 de 4,219 predicciones: 177, 101 y 12 respectivamente. El 93.13% permaneció idéntico. Es una regla aplicada a las decisiones, no una afirmación de que ningún modelo adicional se entrenó. La estrategia mantiene las recetas seleccionadas y reduce el alcance de la modificación final.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/materialize_hailmary_s1_foldbag_router_20260507.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md.

## 9. Resultados del sistema completo (8:30–9:20)

El sistema completo obtuvo 0.60798 de macro-F1 en severidad, tercer lugar de dieciséis equipos. Para tipos obtuvo 0.87016 de micro-F1, sexto de quince. La mejora final sobre el ancla fue pequeña: de 0.60678 a 0.60798, alrededor de 0.00120. La salida de tipos permaneció exactamente igual. Separar esta mejora de la aumentación evita atribuirle al último ajuste toda la historia del proyecto. Los resultados oficiales orientaron la selección, por lo que el test no fue una evaluación final completamente independiente. Estos son resultados de benchmark, no validación en atención social ni evidencia de transferencia entre instituciones.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md; corrected paper, official results. https://www.codabench.org/competitions/14614/

## 10. Una estrategia que se puede poner a prueba en otros contextos (9:20–10:00)

La propuesta que puede generalizarse es una estrategia comprobable: comparar familias con protocolos comunes, adaptar salidas y pérdidas al significado de las etiquetas, intervenir sobre clases escasas y medir todos los errores relevantes. Una vez elegidas las recetas, limitar cambios alrededor de una referencia permite controlar cuántas decisiones se modifican. Para poner a prueba esa estrategia fuera del concurso, proponemos auditar las variantes sintéticas, controlar balance y pesos, repetir semillas y evaluar por institución y nivel de consenso. Estas son las siguientes pruebas; todavía no demostramos generalización entre instituciones. Gracias.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md, limitations and future validation.
