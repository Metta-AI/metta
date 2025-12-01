{{- define "cronjob.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "cronjob.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "cronjob.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "cronjob.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "cronjob.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cronjob.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "cronjob.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "cronjob.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "standard-service-account" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
