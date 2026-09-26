# Pulso TransMi: automatización de entregas

Servicio de inferencia para la competencia académica Pulso TransMi. Recoge
observaciones, carga un modelo versionado de Storage privado, predice y entrega
al ciclo abierto. PostgreSQL/Supabase conserva el checkpoint, los payloads,
recibos y métricas de evaluación. No necesita un computador personal encendido.

Este repositorio publica únicamente el código de automatización y sus pruebas.
No contiene datasets, pesos del modelo, credenciales ni el historial Git del
proyecto original. Código derivado del SDK Pulso TransMi, bajo licencia MIT.

## Ejecución durante la campaña

- Sesiones de hasta cinco horas en runners estándar de GitHub Actions.
- Consulta la API cada dos minutos, también fuera de la ventana aproximada
  :35–:05, porque el horario real puede variar.
- Cada sesión solicita la siguiente por `workflow_dispatch` antes de terminar.
- Cron a los :02, :32 y :47 como respaldo si se pierde una sesión/relevo.
- Una sola sesión activa: el grupo de concurrencia conserva la sesión actual.
- La variable `AUTOMATION_UNTIL` establece el fin UTC de la campaña. Al llegar,
  no se inicia otra sesión y el workflow se deshabilita.
- Los fallos transitorios se reintentan; cinco fallos consecutivos terminan la
  sesión como fallida y solicitan relevo. Una caída prolongada de GitHub/API
  sigue pudiendo causar ventanas perdidas; no existe garantía de 24/24 entregas.

La fecha de fin se configura al activar las dos semanas solicitadas. Para parar
antes: **Actions → Entregas automaticas → Disable workflow**, y cancelar también
la ejecución activa y cualquier pendiente. No basta cancelar una sesión si
permanece habilitado el cron.

## Configuración privada

Secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `PULSO_API_KEY` y
`EXPECTED_PARTICIPANT_ID`. Se comprueba la identidad antes de enviar.
Los logs públicos muestran solo estados y contadores. Los recibos, datos,
predicciones y métricas detalladas permanecen en Supabase, con acceso backend.
No hay workflows `pull_request_target` ni acceso a secretos en las pruebas de PR.

El artefacto activo se verifica por SHA-256 y versiones exactas de dependencias.
El ID persistido de cada envío permite reintentar sin crear otra entrega.
Las métricas locales MAE/RMSE se actualizan tras la inferencia cuando hay datos
reales completos; no se presentan como puntuación oficial.

GitHub ofrece runners estándar gratuitos en repositorios públicos:
https://docs.github.com/en/billing/concepts/product-billing/github-actions
El límite por job no equivale a una garantía de disponibilidad del servicio.

## Pruebas

`python -m pip install -r requirements-operativo.txt pytest`

`python -m pip install --no-deps .`

`python -m pytest -q`
