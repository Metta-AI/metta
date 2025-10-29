{{- define "softmax-cronjob.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "softmax-cronjob.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "softmax-cronjob.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{ include "softmax-cronjob.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "softmax-cronjob.selectorLabels" -}}
app.kubernetes.io/name: {{ include "softmax-cronjob.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "softmax-cronjob.serviceAccountName" -}}
{{- include "softmax-cronjob.fullname" . }}
{{- end }}
