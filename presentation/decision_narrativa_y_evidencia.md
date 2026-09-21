# Decisiones de exposición y revisión de evidencia

Versión del 21 de septiembre de 2026. Charla ejecutiva de diez minutos, con español para revisión e inglés para exposición. Se conserva el requisito del correo de centrarse en implementación y evitar una introducción extensa del corpus. El póster A0 vertical es una propuesta de diseño: el correo no establece dimensiones obligatorias.

## Hilo principal

La exposición muestra cómo adaptar un clasificador al problema: comparar las tres vías exploradas, distinguir severidad ordinal de tipos coexistentes, intervenir sobre la escasez de Severe y evaluar qué errores cambian. El ancla ocupa la parte final, como ajuste de decisiones sobre recetas seleccionadas. No se atribuye toda la contribución al incremento final de puntuación.

## Qué sustenta cada afirmación

1. **Tres vías de modelado.** Código clásico, prompting/LoRA y encoders del proyecto camera-ready de junio. Los LLM también son Transformers; las categorías describen formas de uso.
2. **Comparación.** Los pilotos generativos comparten 133 reportes y exactitud conjunta. La tabla clásicos/BETO usa desarrollo completo y distingue las métricas de S1 y S2. El 0.4974 es una referencia clásica temprana, no su mejor ensamble posterior. Las comparaciones evalúan recetas completas y no aíslan arquitectura ni presupuesto.
3. **Salidas.** Código de la receta S1 con softmax, focal, distancia ordinal y cabeza auxiliar enmascarada. S2 final tiene seis sigmoid y deriva N/A. Esto corrige la descripción general de siete sigmoid que todavía aparece en el paper: el detalle de implementación se verifica en el código.
4. **Aumentación.** La implementación compone plantillas de apertura, acciones, contexto y cierre. Algunas elecciones usan indicios del original. Conserva originales, hereda etiquetas y limita a dos variantes por origen. Añade 516 textos Severe, de 4.52% a 8.00%. No se describe como paráfrasis verificada ni como generación por LLM.
5. **Efecto.** Macro-F1 y F1 Severe suben; recall Severe baja. La gráfica usa diferencias absolutas por cien, eje simétrico y cero explícito. Los valores originales acompañan el gráfico. Balance, estilo y pesos cambian conjuntamente, por lo que falta una ablación causal.
6. **Anotación.** Nuevo análisis de etiquetas suaves, posterior al paper. 877/1,212 reportes tienen algún tipo dividido. La varianza p(1-p) describe votos dentro de cada decisión. Sin identidades, votos S1 ni profesión confirmada no puede presentarse como variación entre trabajadores sociales identificados. Hay 24 discrepancias hard/soft en desarrollo, de causa desconocida.
7. **Ancla.** El router modifica 290/4,219 decisiones, en tres transiciones permitidas. El 93.13% conserva el ancla. Se compara el resultado oficial anterior y final, sin confundirlo con el efecto de aumentación.
8. **Generalización.** Se propone una estrategia transferible como hipótesis de trabajo. No se afirma validación en otras instituciones. El test oficial orientó la selección de entregas.

## Cambios de esta revisión

Se incorporó la comparación directa TF-IDF/SVM frente a BETO, se conservaron diagramas editables de flujos y cabezas, se distinguieron protocolos y se sustituyó el correo personal por la dirección del repositorio. Notas y guiones remiten a rutas de la copia preparada para compartir. Las cifras de anotación se acompañan de script reproducible y resultados agregados.

La revisión del proyecto original confirmó que GitHub era anterior al camera-ready de junio. Los cambios posteriores afectan principalmente paper y documentación; los 26 archivos científicos seleccionados para compartir coinciden entre ambas versiones antes de la limpieza de rutas. Se recuperaron los PDF EN/ES actuales por separado. El repositorio original conserva historial privado y no debe hacerse público directamente.

## Revisión técnica

Las presentaciones contienen diez diapositivas por idioma, gráficos y tablas nativos, diagramas editables y notas del orador. Se verificaron estructura del paquete, datos de gráficos y geometría. La revisión visual usa imágenes renderizadas; no constituye una prueba dentro de PowerPoint. Los PDF se revisan renderizados a imagen. Los materiales y su documentación no incluyen relatos, identificadores de casos ni correos personales.
