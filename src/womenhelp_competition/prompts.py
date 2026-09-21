from __future__ import annotations

from .labels import SUBTASK1_LABELS, SUBTASK2_LABELS

BASE_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Debes leer un reporte narrativo y devolver SOLO un JSON válido, sin explicaciones adicionales.
Usa exclusivamente las etiquetas oficiales del concurso.
"""

EXPERT_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
EVIDENCE_JSON_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Debes leer un reporte narrativo, razonar internamente primero y devolver SOLO un JSON válido.
La salida debe ser compacta, útil para fine-tuning y estrictamente fiel al reporte.
Cada elemento de `evidence` debe contener una cita textual breve del reporte y una justificación breve, precisa y directamente anclada a esa cita.
Usa exclusivamente las etiquetas oficiales del concurso y no inventes hechos, contexto ni agravantes ausentes.
"""
EVIDENCE_JSON_FLOW_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Piensa de forma ordenada, pero devuelve SOLO un JSON válido y compacto.
Tu tarea es imitar un razonamiento cuidadoso: primero ubicar citas textuales, luego decidir etiquetas solo con apoyo textual directo y al final responder con el JSON.
No inventes hechos, no uses explicaciones meta y no justifiques por qué descartaste etiquetas; la evidencia final debe sostener únicamente la respuesta elegida.
"""
EVIDENCE_JSON_FLOW_CALIBRATED_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Piensa de forma ordenada y conservadora, pero devuelve SOLO un JSON válido y compacto.
Tu tarea es imitar un razonamiento del dataset: primero ubicar citas textuales exactas, luego conservar solo etiquetas con apoyo textual directo y al final responder con el JSON.
No inventes hechos, no uses explicaciones meta y, ante conflicto entre una interpretación amplia y el uso observado del dataset, sigue la lectura más literal y estrecha del reporte.
"""
EVIDENCE_JSON_SUPPORT_ONLY_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Debes decidir etiquetas con criterio conservador y luego devolver SOLO un JSON válido.
Cada cita textual en `evidence` debe apoyar de forma positiva al menos una etiqueta incluida en `violence_types`, salvo cuando la respuesta final sea `N/A`.
La salida debe ser breve, fiel al reporte y útil para fine-tuning.
"""
EVIDENCE_JSON_GOLD_GUIDED_SYSTEM_PROMPT = """Eres una analista experta en clasificación de reportes de violencia de género.
Ya conoces la respuesta objetivo correcta del dataset y NO debes modificarla.
Tu tarea es reconstruir un razonamiento inductivo breve: primero ubicar citas textuales exactas y luego explicar cómo esas citas sostienen la respuesta objetivo fija.
La salida final debe ser SOLO un JSON válido, compacto y fiel al reporte.
"""
COMPACT_SYSTEM_PROMPT = """Clasifica reportes de violencia de género.
Devuelve solo un JSON válido con las llaves severity y violence_types.
"""


def build_joint_prompt(text: str) -> str:
    return f"""Clasifica el siguiente reporte en dos dimensiones.

1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}

Reglas:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- si no hay un tipo claro, puedes usar N/A
- si incluyes N/A no mezcles N/A con otros tipos

Esquema de salida:
{{
  "severity": "High",
  "violence_types": ["Psychological", "Physical"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_expert(text: str) -> str:
    return f"""Clasifica el siguiente reporte en dos dimensiones.

1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}

Reglas:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- si no hay un tipo claro, puedes usar N/A
- si incluyes N/A no mezcles N/A con otros tipos

Reglas expertas adicionales:
- decide cada etiqueta solo si hay evidencia textual suficiente en el reporte
- `Property-related` solo si hay daño, despojo, retención o afectación real de bienes, documentos o recursos; romper objetos para intimidar cuenta como `Psychological`
- `Vicarious` solo si hijas, hijos o personas cercanas son usados como instrumento para dañar a la mujer; una agresión reactiva al menor no basta por sí sola
- `N/A` solo aplica si no hay violencia explícita o el caso es puramente administrativo
- la severidad debe reflejar el acto explícito más grave del caso completo

Esquema de salida:
{{
  "severity": "High",
  "violence_types": ["Psychological", "Physical"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_compact(text: str) -> str:
  return f"""Devuelve solo JSON válido.

severity: una de {', '.join(SUBTASK1_LABELS)}
violence_types: cero o más de {', '.join(SUBTASK2_LABELS)}

Reglas:
- usa exactamente esas etiquetas
- no escribas explicación
- si no hay tipo claro usa N/A
- no mezcles N/A con otros tipos

Formato:
{{"severity":"High","violence_types":["Psychological","Physical"]}}

Reporte:
{text}
"""


def build_joint_prompt_evidence_json(text: str) -> str:
  return f"""Clasifica el siguiente reporte en dos dimensiones y agrega evidencia breve.

1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}
3. evidence: una lista de 1 a 4 objetos con quote y reason

Reglas:
- piensa primero en silencio y luego devuelve solo el JSON final
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- cada `quote` debe ser una cita textual breve copiada del reporte, no una paráfrasis
- cada `reason` debe ser una justificación corta, directa y fiel que explique qué etiqueta o etiquetas sustenta esa cita
- procura que la evidencia cubra todos los tipos de violencia que declares en `violence_types`
- la severidad debe seguir un criterio conservador y ser la mínima categoría compatible con la evidencia explícita del reporte
- ante duda, elige la categoría menor compatible con el texto
- no subas la severidad por intuición, por consecuencias emocionales inferidas, por contexto administrativo o por un juicio moral del caso
- en este esquema, insultos, celos, incumplimiento económico, conflictos legales, retención de convivencia, cachetadas, empujones, jalones o amenazas verbales aisladas suelen quedarse en `Mild` si el reporte no menciona agravantes mayores
- usa `Medium` solo si el texto describe explícitamente escalamiento claro, repetición severa o múltiples actos graves en el mismo caso
- usa `High` solo si hay evidencia explícita de riesgo extremo, arma, lesión grave, violencia sexual consumada o peligro serio inmediato
- usa `Severe` solo en casos excepcionalmente graves y explícitos
- `Psychological` requiere insultos, humillación, amenazas, intimidación, control o violencia psicológica explícita; no la infieras solo por tristeza, terapia, infidelidad, separación o malestar
- `Economic` requiere privación, control, retención o incumplimiento económico explícito; no la infieras solo por problemas económicos generales o por tramitar pensión
- `Vicarious` solo aplica si hijas, hijos o personas cercanas son usados explícitamente para dañar o controlar a la mujer; accidentes, omisiones o conflictos de cuidado por sí solos no bastan
- `Property-related` solo aplica si el texto describe daño, despojo, retención o afectación concreta de bienes, documentos o vivienda; una amenaza verbal aislada no basta por sí sola
- si no hay un tipo claro, puedes usar N/A
- si incluyes N/A no mezcles N/A con otros tipos

Esquema de salida:
{{
  "evidence": [
    {{"quote": "me amenaza con quitarme a mi hija", "reason": "Sustenta violencia psicológica y vicaria"}},
    {{"quote": "me limita el dinero", "reason": "Sustenta violencia económica"}}
  ],
  "severity": "High",
  "violence_types": ["Psychological", "Economic", "Vicarious"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_evidence_json_flow(text: str) -> str:
  return f"""Clasifica el siguiente reporte en dos dimensiones y agrega evidencia breve.

Haz internamente este flujo y luego devuelve SOLO el JSON final:
1. ubica de 1 a 4 citas textuales exactas del reporte
2. decide qué etiqueta sustenta directamente cada cita
3. elimina cualquier etiqueta sin apoyo textual directo
4. elige la severidad mínima compatible con el acto explícito más grave
5. responde solo con el JSON

Objetivo:
1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}
3. evidence: una lista de 1 a 4 objetos con quote y reason

Reglas de evidencia:
- cada `quote` debe ser una cita textual breve copiada literalmente del reporte
- cada `reason` debe ser una justificación corta y positiva que explique qué etiqueta o etiquetas sí sustenta esa cita
- no uses `evidence` para explicar por qué descartaste una etiqueta
- si la respuesta final no es `N/A`, cada elemento de `evidence` debe apoyar al menos una etiqueta elegida en `violence_types`
- si la respuesta final es `N/A`, usa una cita que muestre ausencia explícita de violencia o que el caso es meramente administrativo

Reglas de decisión:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- procura que la evidencia cubra todos los tipos de violencia que declares en `violence_types`
- no inventes hechos, contexto ni agravantes ausentes
- decide cada etiqueta según el uso real del dataset, no según una explicación legal abstracta
- la severidad debe seguir un criterio muy conservador y ser la mínima categoría compatible con la evidencia explícita del reporte
- ante duda, elige la categoría menor compatible con el texto

Guía por tipo:
- `Psychological`: insultos, humillación, amenazas, intimidación, críticas constantes, control, chantaje, violencia psicológica explícita o daño psicológico atribuido por la usuaria al actuar de la pareja o agresor
- `Economic`: falta de apoyo, incumplimiento de pensión, control o limitación del dinero, deslinde económico, privación de recursos o gastos del hogar o menores asumidos por la usuaria ante incumplimiento del agresor
- `Physical`: golpes, cachetadas, empujones, jalones, estrujones, forcejeos o cualquier agresión corporal explícita
- `Sexual`: tocamientos, coerción, abuso o violencia sexual explícita
- `Property-related`: daño, despojo, retención o afectación concreta de bienes, documentos, vivienda o recursos materiales
- `Vicarious`: uso explícito de hijas, hijos o personas cercanas como instrumento para dañar o controlar a la mujer
- `N/A`: ausencia explícita de violencia o caso puramente administrativo

Reglas de calibración del dataset:
- `problemas económicos`, `discusiones por dinero` o `tramitar pensión` por sí solos no bastan para `Economic`
- `ha vivido violencia psicológica`, `daños psicológicos`, críticas, chantajes o amenazas sí sostienen `Psychological`
- infidelidad, conversaciones o imágenes de otras mujeres y abandono afectivo pueden sostener `Psychological` si el reporte los presenta como agravio o daño en la relación
- chantajes sobre convivencia, custodia o ver a menores suelen quedarse en `Psychological`; usa `Vicarious` solo si el texto muestra que hijas o hijos son usados explícitamente para dañar o controlar a la usuaria
- referencias a nietos normalmente no bastan para `Vicarious`
- una cachetada, jalón, empujón, forcejeo o amenaza verbal aislada suele quedarse en `Mild` si no hay agravantes mayores explícitos
- usa `Medium` solo si el texto describe varias agresiones relevantes en el mismo caso, repetición severa o escalamiento claro
- usa `High` solo si hay evidencia explícita de riesgo extremo, arma, lesión grave, violencia sexual consumada o peligro serio inmediato
- usa `Severe` solo en casos excepcionalmente graves y explícitos

Esquema de salida:
{{
  "evidence": [
    {{"quote": "me dijo que si me iba me quitaría a mis hijos", "reason": "Sustenta violencia psicológica por amenaza y control"}},
    {{"quote": "no me apoya con los gastos de mis hijos", "reason": "Sustenta violencia económica por incumplimiento de apoyo"}}
  ],
  "severity": "Mild",
  "violence_types": ["Psychological", "Economic"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_evidence_json_flow_calibrated(text: str) -> str:
  return f"""Clasifica el siguiente reporte en dos dimensiones y agrega evidencia breve.

Haz internamente este flujo y luego devuelve SOLO el JSON final:
1. ubica de 1 a 4 citas textuales exactas del reporte
2. asigna a cada cita solo las etiquetas que sí estén sostenidas por esa cita
3. elimina cualquier etiqueta sin apoyo textual directo
4. elige la severidad mínima compatible con el acto explícito más grave
5. responde solo con el JSON

Objetivo:
1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}
3. evidence: una lista de 1 a 4 objetos con quote y reason

Reglas de evidencia:
- cada `quote` debe ser una cita textual breve copiada literalmente del reporte
- cada `reason` debe ser una justificación corta y positiva que explique qué etiqueta o etiquetas sí sustenta esa cita
- no uses `evidence` para explicar por qué descartaste una etiqueta
- si la respuesta final no es `N/A`, cada elemento de `evidence` debe apoyar al menos una etiqueta elegida en `violence_types`
- si la respuesta final es `N/A`, usa una cita que muestre ausencia explícita de violencia o que el caso es meramente administrativo

Reglas de decisión:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- procura que la evidencia cubra todos los tipos de violencia que declares en `violence_types`
- no inventes hechos, contexto ni agravantes ausentes
- decide cada etiqueta según el uso real del dataset, no según una explicación legal abstracta
- la severidad debe seguir un criterio muy conservador y ser la mínima categoría compatible con la evidencia explícita del reporte
- no subas la severidad solo porque haya varias etiquetas, varias citas o varios detalles dentro del mismo episodio
- ante duda, elige la categoría menor compatible con el texto

Guía por tipo:
- `Psychological`: insultos, humillación, amenazas, intimidación, críticas constantes, control, chantaje, violencia psicológica explícita o daño psicológico atribuido por la usuaria al actuar de la pareja o agresor
- `Economic`: falta de apoyo, incumplimiento de pensión, control o limitación del dinero, deslinde económico, privación de recursos o gastos del hogar o menores asumidos por la usuaria ante incumplimiento del agresor
- `Physical`: golpes, cachetadas, empujones, jalones, estrujones, forcejeos o cualquier agresión corporal explícita
- `Sexual`: tocamientos, coerción, abuso o violencia sexual explícita
- `Property-related`: daño, despojo, retención o afectación concreta de bienes, documentos, vivienda o recursos materiales
- `Vicarious`: uso explícito y central de hijas o hijos de la usuaria como instrumento para dañarla o controlarla
- `N/A`: ausencia explícita de violencia o caso puramente administrativo

Reglas de calibración del dataset:
- `problemas económicos`, `discusiones por dinero`, separación o `tramitar pensión` por sí solos no bastan para `Economic`
- `ha vivido violencia psicológica`, `daños psicológicos`, críticas, chantajes o amenazas sí sostienen `Psychological`
- infidelidad, conversaciones o imágenes de otras mujeres y abandono afectivo pueden sostener `Psychological` si el reporte los presenta como agravio o daño en la relación
- si el reporte dice que la usuaria cubre sola los gastos, mantiene a hijas o hijos, o que el agresor se deslinda del apoyo, eso sí puede sostener `Economic`
- chantajes sobre convivencia, custodia o ver a menores suelen quedarse en `Psychological`; usa `Vicarious` solo si hijas o hijos de la usuaria son usados explícitamente como instrumento directo para dañarla o controlarla
- amenazas sobre nietos, nueras, parejas, familiares o sobre `dejarla sin ver a sus nietos` no bastan para `Vicarious`; normalmente se quedan en `Psychological`
- referencias a nietos normalmente no bastan para `Vicarious`
- una cachetada, jalón, empujón, forcejeo o amenaza verbal aislada suele quedarse en `Mild` aunque aparezca junto con chantaje o discusión, si no hay agravantes mayores explícitos
- usa `Medium` solo si el texto describe varias agresiones relevantes en el mismo caso con repetición severa, escalamiento claro o daño importante explícito
- usa `High` solo si hay evidencia explícita de riesgo extremo, arma, lesión grave, violencia sexual consumada o peligro serio inmediato
- usa `Severe` solo en casos excepcionalmente graves y explícitos

Esquema de salida:
{{
  "evidence": [
    {{"quote": "me dijo que si me iba me quitaría a mis hijos", "reason": "Sustenta violencia psicológica por amenaza y control"}},
    {{"quote": "no me apoya con los gastos de mis hijos", "reason": "Sustenta violencia económica por incumplimiento de apoyo"}}
  ],
  "severity": "Mild",
  "violence_types": ["Psychological", "Economic"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_evidence_json_support_only(text: str) -> str:
  return f"""Clasifica el siguiente reporte y devuelve solo un JSON final compacto.

Tu razonamiento interno debe seguir este orden:
1. encuentra citas textuales exactas
2. asigna a cada cita solo las etiquetas que sí estén sostenidas por esa cita
3. conserva solo etiquetas con apoyo positivo
4. fija la severidad más baja compatible con el hecho explícito más grave

Salida objetivo:
1. severity: exactamente una de estas etiquetas: {', '.join(SUBTASK1_LABELS)}
2. violence_types: cero o más etiquetas de esta lista: {', '.join(SUBTASK2_LABELS)}
3. evidence: una lista de 1 a 4 objetos con quote y reason

Restricciones fuertes:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- usa exactamente los nombres de etiqueta indicados
- no escribas razones negativas como `no sustenta`, `no describe`, `no basta` o explicaciones sobre etiquetas descartadas
- cada `quote` debe ser textual y breve
- cada `reason` debe nombrar al menos una etiqueta elegida en la respuesta final
- no mezcles `N/A` con otros tipos

Atajos de calibración:
- `Psychological` sí aplica ante violencia psicológica declarada, chantaje, amenaza, insulto, humillación, control, crítica constante o daño psicológico explícito
- `Economic` sí aplica ante falta de apoyo, incumplimiento de gasto o pensión, deslinde económico o limitación material explícita
- `Economic` no aplica por mera mención de `problemas económicos` o por el simple hecho de pedir asesoría de pensión
- `Vicarious` exige uso explícito de hijas o hijos para dañar o controlar a la usuaria; disputas sobre nietos o convivencia suelen quedarse en `Psychological`
- `Mild` es la opción por defecto salvo agravantes muy explícitos

Esquema de salida:
{{
  "evidence": [
    {{"quote": "me grita y me amenaza", "reason": "Sustenta violencia psicológica por amenaza explícita"}}
  ],
  "severity": "Mild",
  "violence_types": ["Psychological"]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_joint_prompt_evidence_json_gold_guided(text: str, gold_severity: str, gold_violence_types: list[str]) -> str:
  gold_types_literal = ", ".join(f'"{label}"' for label in gold_violence_types)
  return f"""Debes producir evidencia y justificación para una respuesta objetivo YA FIJADA.

Respuesta objetivo fija del dataset:
- severity: {gold_severity}
- violence_types: [{gold_types_literal}]

Tu trabajo NO es reclasificar el caso.
Debes actuar como si reconstruyeras el camino inductivo hacia esa respuesta:
1. ubica de 1 a 4 citas textuales exactas del reporte
2. asigna a cada cita solo etiquetas incluidas en la respuesta objetivo
3. redacta razones positivas, breves y ancladas a la cita
4. devuelve el JSON final conservando EXACTAMENTE la respuesta objetivo

Reglas obligatorias:
- responde solo con JSON válido
- no agregues texto fuera del JSON
- `severity` debe ser exactamente `{gold_severity}`
- `violence_types` debe ser exactamente [{gold_types_literal}] y en ese mismo orden
- no agregues, elimines, sustituyas ni reordenes etiquetas
- cada `quote` debe ser una cita textual breve copiada literalmente del reporte
- cada `reason` debe ser una justificación positiva que apoye una o más etiquetas de la respuesta objetivo
- no uses razones negativas, metaexplicaciones ni explicaciones sobre etiquetas descartadas
- si hay varias etiquetas en la respuesta objetivo, procura que la evidencia cubra todas
- no inventes hechos, contexto ni agravantes ausentes

Esquema de salida:
{{
  "evidence": [
    {{"quote": "cita textual exacta", "reason": "Justificación breve y positiva anclada a la cita"}}
  ],
  "severity": "{gold_severity}",
  "violence_types": [{gold_types_literal}]
}}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_severity_prompt(text: str) -> str:
    return f"""Clasifica la severidad del siguiente reporte.

Devuelve solo JSON válido con este esquema:
{{
  "severity": "High"
}}

Etiquetas permitidas: {', '.join(SUBTASK1_LABELS)}

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""


def build_types_prompt(text: str) -> str:
    return f"""Clasifica los tipos de violencia del siguiente reporte.

Devuelve solo JSON válido con este esquema:
{{
  "violence_types": ["Psychological", "Physical"]
}}

Etiquetas permitidas: {', '.join(SUBTASK2_LABELS)}

Reglas:
- usa cero o más etiquetas
- si no hay un tipo claro, devuelve ["N/A"]
- si incluyes N/A no mezcles N/A con otros tipos

Reporte:
<<<REPORTE>>>
{text}
<<<FIN_REPORTE>>>
"""
