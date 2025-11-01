{{- define "pr-similarity-cache-cronjob.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "pr-similarity-cache-cronjob.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "pr-similarity-cache-cronjob.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "pr-similarity-cache-cronjob.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "pr-similarity-cache-cronjob.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pr-similarity-cache-cronjob.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "pr-similarity-cache-cronjob.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- include "pr-similarity-cache-cronjob.fullname" . }}
{{- else -}}
{{- .Values.serviceAccount.name | default (include "pr-similarity-cache-cronjob.fullname" .) }}
{{- end -}}
{{- end }}
