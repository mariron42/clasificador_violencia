# Guion de exposición

10 diapositivas principales para 10 minutos y dos diapositivas finales de respaldo para preguntas.

## 1. Clasificación de reportes de violencia (0:00–0:30)

Buenos días. Soy Marcel Herrera Rendón, de la Universidad de Sonora.

Voy a contar cómo construimos nuestro sistema para WomenHelp. Partimos comparando distintas formas de clasificar reportes y fuimos adaptando el entrenamiento a dos tareas: severidad y tipos de violencia.

La historia incluye un aprendizaje importante: mejorar en desarrollo no siempre significó mejorar en el test. Eso influyó en cómo aprovechamos la aumentación de datos y en la forma del sistema final.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md (corrected system-description paper).

Respaldo para preguntas (no leer como parte del guion): Buenos días. Soy Marcel Herrera Rendón, de la Universidad de Sonora. Voy a contar la estrategia que seguimos para clasificar reportes de violencia en WomenHelp. El punto central es cómo el problema orientó la elección de modelos, las salidas y los datos de entrenamiento. Después de encontrar recetas útiles, limitamos los cambios finales alrededor de una referencia. El corpus ya se explica en otra sesión, por lo que me concentraré en la implementación.

## 2. Tres familias, tres formas de clasificar (0:30–1:30)

Exploramos métodos clásicos, modelos generativos y encoders Transformer.

Los clásicos representan palabras y caracteres mediante TF-IDF y después clasifican. Los generativos reciben instrucciones y producen las etiquetas como una respuesta estructurada. Los encoders construyen una representación contextual que conectamos con salidas específicas de clasificación.

Los LLM también utilizan Transformers; aquí distinguimos cómo resolvemos la tarea. Para la vía generativa teníamos que comprobar tanto el formato de la respuesta como sus etiquetas. Los encoders nos permitían controlar directamente las salidas y las pérdidas.

La comparación nos ayudó a decidir dónde concentrar el trabajo.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md, model families and implementation.

Respaldo para preguntas (no leer como parte del guion): Exploramos tres familias. Los modelos clásicos convierten palabras y caracteres en vectores TF-IDF y después clasifican. Los LLM generativos reciben instrucciones y producen una respuesta estructurada; probamos prompting y adaptación LoRA. Los encoders Transformer, como BETO, aprenden una representación contextual y alimentan cabezas de clasificación. Los LLM también usan Transformers: la distinción aquí es su forma de uso, generativa frente a discriminativa. Los clásicos dieron referencias fuertes y diversidad; los encoders permitieron controlar las salidas. La vía generativa quedó fuera del sistema final. La elección respondió a nuestros experimentos y recursos, no a una superioridad universal de una familia.

## 3. La evidencia orientó la elección (1:30–2:20)

En el piloto de 133 reportes, TF-IDF con SGD obtuvo mejor exactitud conjunta que los pilotos generativos que mostramos. Esa métrica exige acertar la severidad y el conjunto completo de tipos.

Hubo un adaptador LoRA temprano que produjo salidas vacías, pero los pilotos posteriores sí generaron respuestas. Por eso no resumimos toda esa vía como un fallo de formato: también comparamos su desempeño de clasificación.

En desarrollo completo, configuraciones posteriores de BETO superaron las referencias clásicas mostradas. Conservamos los clásicos como referencias y componentes del ensamble, y profundizamos en los encoders especializados.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md, classical/encoder and common 133-record comparisons.

Respaldo para preguntas (no leer como parte del guion): A la izquierda comparo exactamente los mismos 133 reportes. La exactitud conjunta exige acertar severidad y todos los tipos: TF-IDF con SGD obtuvo 0.2331, el piloto Gemma con LoRA 0.1955 y Qwen con prompting 0.1053. El piloto LoRA usó 625 ejemplos y 80 pasos. No representa el potencial de todos los LLM. A la derecha comparo recetas clásicas y BETO en desarrollo completo. En severidad, la referencia clásica temprana dio 0.4974 y BETO focal 0.5429 de macro-F1. Hubo ensambles clásicos posteriores de 0.5047 y 0.5215. En tipos, el SVM de caracteres dio 0.7642 de macro-F1 y 0.8449 de micro-F1; BETO con umbral fijo 0.45 dio 0.8198 y 0.8685. Son recetas distintas, no una ablación aislada de arquitectura. No mezclo ambas escalas porque cambian los datos y la métrica. La conclusión práctica fue conservar referencias clásicas y profundizar en encoders especializados.

## 4. El problema determina las salidas (2:20–3:35)

Severidad requiere elegir una categoría entre cuatro niveles ordenados. Los tipos de violencia pueden coexistir: económica, física, patrimonial, psicológica, sexual y vicaria.

En la receta de severidad que mostramos, normalizamos espacios, tokenizamos y limitamos la entrada a 512 tokens. La salida principal utiliza focal ponderada y una penalización ordinal. Una cabeza auxiliar aprende información de tipos donde esas etiquetas están disponibles.

Esa condición importa porque las subtareas no comparten todos sus registros. La ausencia de una etiqueta auxiliar no se interpreta como un negativo.

El clasificador final de tipos es otro BETO: procesa hasta 384 tokens y tiene seis salidas independientes. Si ninguna alcanza 0.45, deriva N/A. Así ajustamos la estructura de las predicciones al significado de cada tarea.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/run_competitive_encoder_severity.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/run_subtask2_encoder_20260429.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md.

Respaldo para preguntas (no leer como parte del guion): Severidad requiere una sola categoría ordenada: Mild, Medium, High o Severe. Los tipos pueden coexistir: económica, física, patrimonial, psicológica, sexual y vicaria. El diagrama superior muestra la receta BETO aumentada: normalizamos espacios, tokenizamos y truncamos a 512 tokens. La cabeza principal usa softmax, focal ponderada y una penalización ordinal de 0.05. Una cabeza auxiliar aprende siete etiquetas de tipos, incluida N/A, con etiquetas suaves y peso 0.3, solo donde existen. Esta máscara evita interpretar información ausente como negativa. El componente final de tipos es otro BETO: procesa hasta 384 tokens, tiene seis sigmoid y umbral 0.45, y deriva N/A cuando ninguna alcanza el umbral. El código confirma seis salidas, aunque una descripción general del paper habla de siete. Estas decisiones adaptan arquitecturas existentes al problema; no proponemos una arquitectura nueva.

## 5. Aumentación dirigida a Severe (3:35–5:05)

Severe tenía 616 ejemplos entre 13,630 reportes de entrenamiento, aproximadamente el 4.5%.

Para aumentar su presencia compusimos textos mediante plantillas de apertura, acciones, contexto y cierre. Parte de las elecciones estaba condicionada por indicios del reporte original. El procedimiento utilizaba una semilla estable y permitía como máximo dos variantes por origen.

Conservamos los originales y las variantes heredaron sus etiquetas. Añadimos 516 textos y Severe pasó a representar el 8% del entrenamiento. Fue un balance parcial.

La evaluación mantuvo los reportes originales. Esto permitía observar qué cambiaba en el modelo después de modificar su entrenamiento. La pregunta siguiente fue cómo se movían sus errores, especialmente en Severe.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/src/womenhelp_competition/text_augmentation.py; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/evidence.json.

Respaldo para preguntas (no leer como parte del guion): Severe tenía 616 de 13,630 ejemplos, apenas 4.52%. Elegimos aumentar solo esa clase en entrenamiento. El método destacado no utiliza un LLM: combina plantillas de apertura, acciones, contexto y cierre con estilo de reporte institucional. Los indicios del original, como relación, horario o referencias a amenazas, condicionan parte de las elecciones. La semilla es estable por registro y generamos como máximo dos variantes por origen. Retenemos los originales y heredamos sus etiquetas. Añadimos 516 textos: ahora Severe tiene 1,132 de 14,146, es decir, 8%. Reducimos el desequilibrio, sin igualar las cuatro clases. Las plantillas pueden introducir hechos, por lo que estos textos no deben presentarse como paráfrasis verificadas. El proyecto también exploró sinónimos y variantes de Mild; la gráfica siguiente corresponde específicamente a las plantillas Severe.

## 6. La mejora incluye un intercambio de errores (5:05–6:15)

Sobre los mismos 3,405 reportes de desarrollo, el macro-F1 mejoró 0.71 puntos y el F1 de Severe mejoró 1.57 puntos, en la escala de diferencias que muestra la gráfica.

Al mismo tiempo, el recall Severe bajó 2.60 puntos. El modelo detectaba menos casos Severe, aunque aumentaba la precisión de las predicciones que hacía para esa clase.

Por eso conservamos varias métricas y seguimos estudiando combinaciones. La receta aumentada era una fuente candidata, no la solución final por sí sola. Además, los pesos de clase se recalcularon con el nuevo entrenamiento: el resultado corresponde a la receta completa.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/evidence.json; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md. Delta = 100*(augmented - baseline).

Respaldo para preguntas (no leer como parte del guion): Para que se vea el efecto, la gráfica muestra cambios absolutos multiplicados por cien y conserva el cero: 0.71 puntos de macro-F1, 1.57 de F1 Severe y menos 2.60 de recall Severe. Al lado están los valores originales en escala cero a uno. El conjunto de desarrollo contiene los mismos 3,405 reportes, sin aumentación. La mejora en F1 no implica recuperar más casos Severe: el recall bajó. Además, al aumentar datos cambiaron las proporciones y los pesos de clase recalculados. No podemos adjudicar el resultado exclusivamente al estilo de las plantillas. Faltan un control de sobremuestreo equivalente, pesos controlados, varias semillas y una auditoría de etiquetas. Esta es evidencia de una receta concreta, no una garantía de mejora general.

## 7. Las decisiones también dependieron del test (6:15–7:15)

Las dos subtareas siguieron trayectorias distintas.

En tipos, sustituir el componente previo por un BETO dedicado produjo una mejora oficial clara: el micro-F1 pasó de aproximadamente 0.849 a 0.870. Conservamos ese componente.

En severidad, varios candidatos prometedores en desarrollo rindieron peor en el test. Una variante ordinal más conservadora, highpen, obtuvo aproximadamente 0.601 de macro-F1 oficial frente al 0.607 del ancla.

También encontramos candidatos cuyo resultado de búsqueda no se reproducía al reconstruir las predicciones. Verificamos esa fidelidad antes de seguir comparando.

La decisión fue mantener la referencia que ya funcionaba y estudiar qué cambios concretos aceptar de una alternativa, en vez de sustituir todas sus predicciones.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/main/docs/final_selection.md; https://github.com/mariron42/clasificador_violencia/blob/main/results/final_development.json

Respaldo para preguntas (no leer como parte del guion): La tabla compara resultados oficiales dentro de cada subtarea; no compara macro S1 con micro S2. highpen es distinto del candidato repro_diverse10 que alcanzó 0.580536 en desarrollo. No se encontró resultado oficial de ese candidato exacto. Otros resultados negativos S1 fueron gated 0.591507, ordinal random10_0008 0.591950 y metaordinal 0.527890. La evidencia no permite atribuir todo el descenso a una única causa ni convertir la fidelidad de reconstrucción en una garantía de generalización.

## 8. El ancla limita los cambios finales (7:15–8:30)

El candidato final combinaba cinco folds BETO con aumentación y cinco Electricidad. Esa combinación proponía una clasificación alternativa a la del ancla Top10.

El router aceptaba únicamente tres transiciones: Medium a Mild, Mild a Medium y High a Severe. Para esta última pedía además indicios léxicos severos. En los demás casos conservaba el ancla.

En el test cambiaron 290 de 4,219 predicciones. La regla conservó todas las predicciones Severe del ancla y añadió doce. Es una propiedad verificable del ZIP final.

En desarrollo sí conocemos las etiquetas verdaderas: el recall Severe pasó de 40.91% con el ancla a 43.51% con el router. Conservó 63 aciertos y añadió cuatro. También añadió falsos positivos, por lo que la precisión bajó.

Este es el resultado del sistema final en desarrollo, distinto del experimento con un modelo aumentado individual.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/main/docs/final_selection.md; https://github.com/mariron42/clasificador_violencia/blob/main/results/final_development.json

Respaldo para preguntas (no leer como parte del guion): Al final ya teníamos recetas útiles. En lugar de volver a entrenarlas para cada ajuste, fijamos como ancla las predicciones de la mejor entrega previa. Esa referencia procedía de un ensamble Top10 con tres BETO, tres Electricidad, tres componentes clásicos y un BERTin. Un candidato de cinco folds BETO aumentado y cinco Electricidad propone cambios mediante umbrales ordinales. El router solo admite Medium a Mild, Mild a Medium y High a Severe. Este último necesita además indicios léxicos severos. Si la propuesta no cumple la regla, se conserva el ancla. Cambiaron 290 de 4,219 predicciones: 177, 101 y 12 respectivamente. El 93.13% permaneció idéntico. Es una regla aplicada a las decisiones, no una afirmación de que ningún modelo adicional se entrenó. La estrategia mantiene las recetas seleccionadas y reduce el alcance de la modificación final.

## 9. Resultados del sistema completo (8:30–9:20)

La entrega final alcanzó aproximadamente 0.608 de macro-F1 en severidad, tercer lugar entre dieciséis equipos, y 0.870 de micro-F1 en tipos, sexto de quince.

El ajuste final de severidad fue pequeño: alrededor de 0.0012 de macro-F1 respecto del ancla. Las predicciones de tipos permanecieron exactamente iguales.

Las puntuaciones oficiales orientaron nuestra selección de entregas. El resultado muestra el comportamiento del sistema completo dentro del benchmark. No permite asignar toda esa ganancia a la aumentación ni calcular el recall del test sin sus etiquetas verdaderas.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/results.md; corrected paper, official results. https://www.codabench.org/competitions/14614/

Respaldo para preguntas (no leer como parte del guion): El sistema completo obtuvo 0.60798 de macro-F1 en severidad, tercer lugar de dieciséis equipos. Para tipos obtuvo 0.87016 de micro-F1, sexto de quince. La mejora final sobre el ancla fue pequeña: de 0.60678 a 0.60798, alrededor de 0.00120. La salida de tipos permaneció exactamente igual. Separar esta mejora de la aumentación evita atribuirle al último ajuste toda la historia del proyecto. Los resultados oficiales orientaron la selección, por lo que el test no fue una evaluación final completamente independiente. Estos son resultados de benchmark, no validación en atención social ni evidencia de transferencia entre instituciones.

## 10. Una estrategia que se puede poner a prueba en otros contextos (9:20–10:00)

El aprendizaje que queremos llevar a otros problemas es un procedimiento: comparar alternativas, adaptar las salidas a las etiquetas y evaluar cómo cambian los errores cuando intervenimos sobre los datos.

También aprendimos a comprobar que un candidato se reconstruye fielmente y a distinguir una mejora local de una mejora en el test.

La aumentación produjo nuevas variantes. La validación y las reglas de combinación determinaron cómo incorporarlas al sistema final.

El siguiente paso es evaluar esa estrategia en otras instituciones y con controles más específicos de la aumentación. El código y la evidencia agregada están disponibles en el repositorio de la presentación.

Muchas gracias.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/method.md, limitations and future validation.

Respaldo para preguntas (no leer como parte del guion): La propuesta que puede generalizarse es una estrategia comprobable: comparar familias con protocolos comunes, adaptar salidas y pérdidas al significado de las etiquetas, intervenir sobre clases escasas y medir todos los errores relevantes. Una vez elegidas las recetas, limitar cambios alrededor de una referencia permite controlar cuántas decisiones se modifican. Para poner a prueba esa estrategia fuera del concurso, proponemos auditar las variantes sintéticas, controlar balance y pesos, repetir semillas y evaluar por institución y nivel de consenso. Estas son las siguientes pruebas; todavía no demostramos generalización entre instituciones. Gracias.

## 11. Severe en el sistema finalmente enviado (Respaldo / Q&A: 45–60 s)

Aquí comparamos el sistema que realmente produjo la entrega final sobre los mismos 3,405 reportes de desarrollo. El ancla Top10 obtuvo 40.91% de recall Severe; el foldbag aumentado con Electricidad, 40.26%; el ancla con router, 43.51%. El router conservó 63 aciertos y añadió cuatro, junto con nueve falsos positivos. Por eso la precisión bajó de 58.88% a 55.83%, aunque el F1 Severe subió a 0.4891. En el ZIP del test conservó las 116 predicciones Severe del ancla y añadió doce. Esto impide una caída de recall respecto del ancla sobre las mismas etiquetas verdaderas, pero el recall absoluto del test y la corrección de esas doce adiciones son desconocidos.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/main/docs/final_selection.md; https://github.com/mariron42/clasificador_violencia/blob/main/results/final_development.json

Respaldo para preguntas (no leer como parte del guion): Resultados recalculados desde devel_verbose.csv y cotejados con packaging_summary.json. Macro-F1: ancla 0.5588786159649086; foldbag 0.5645633048288119; router 0.5668988893050375. Recall Severe: 63/154, 62/154, 67/154. Predicciones Severe: 107, 106, 120. El micro-F1 del router recalculado es 0.5882525697503671, concordante con el JSON; el reporte narrativo dice 0.588840. El router se seleccionó en devel. La conservación de Severe es una propiedad de la regla seleccionada y del ZIP, no una restricción universal de todos los routers explorados ni una garantía sobre precisión u otros costos ordinales. El ensamble intermedio de mayo 5 y el score de búsqueda 41.56% no explican esta entrega.

## 12. El desacuerdo permanece en las etiquetas (Respaldo / Q&A: 45–60 s)

Este análisis se añadió después del paper y utiliza las etiquetas suaves de tipos. Cada proporción resume tres votos. En desarrollo, 877 de 1,212 reportes, un 72.36%, tienen al menos uno de los seis tipos con votos divididos. La gráfica muestra la proporción sobre todos los reportes para cada tipo. En una decisión con dos votos positivos y uno negativo, p es dos tercios y la varianza interna es dos novenos. Estas etiquetas conservan información sobre consenso que una etiqueta dura pierde. La cabeza auxiliar de la receta S1 aprovechó esa información donde estaba disponible.

Fuente: https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/docs/annotation_consensus.md; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/results/annotation_consensus.json; https://github.com/mariron42/clasificador_violencia/blob/92aa2ad0a8568dc729da674bcc81040d01f64fa4/scripts/analyze_annotations.py. New analysis after the paper.

Respaldo para preguntas (no leer como parte del guion): Añadí este análisis después del envío del paper. Las etiquetas suaves de tipos toman valores cero, un tercio, dos tercios y uno: conservan la fracción de tres votos positivos. En desarrollo, 877 de 1,212 reportes, 72.36%, tienen al menos uno de los seis tipos con voto dividido. Psicológica registra 33% de decisiones divididas y sexual 6.44%, siempre sobre todos los reportes. Las prevalencias influyen, así que esta comparación no mide directamente la dificultad de cada categoría. La varianza dentro de una decisión binaria es p por uno menos p: cero en unanimidad y dos novenos en división. No tenemos identidades para comparar trabajadores, ni votos de severidad, y la fuente consultada solo identifica anotadores humanos. Encontramos además 24 discrepancias entre mayoría suave y etiqueta dura en desarrollo, cuya causa no conocemos. Esta evidencia invita a evaluar por consenso. La cabeza auxiliar S1 sí aprovechó etiquetas suaves, pero el componente S2 final usó duras; los experimentos S2 con suaves fueron posteriores al envío.
