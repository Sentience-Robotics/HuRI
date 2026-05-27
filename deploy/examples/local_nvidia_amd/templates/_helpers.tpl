{{/*
Expand the name of the chart.
*/}}
{{- define "huri.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully-qualified app name.
Truncated at 63 chars because some Kubernetes name fields have this limit.
*/}}
{{- define "huri.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label: <name>-<version>.
*/}}
{{- define "huri.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels applied to every resource.
*/}}
{{- define "huri.labels" -}}
helm.sh/chart: {{ include "huri.chart" . }}
{{ include "huri.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels (used in matchLabels / ingress backends).
*/}}
{{- define "huri.selectorLabels" -}}
app.kubernetes.io/name: {{ include "huri.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the KubeRay-managed serve service.
KubeRay appends "-serve-svc" to the RayService name.
*/}}
{{- define "huri.serveSvcName" -}}
{{- printf "%s-serve-svc" (include "huri.fullname" .) }}
{{- end }}

{{/*
Name of the KubeRay-managed head service.
KubeRay appends "-head-svc" to the RayService name.
*/}}
{{- define "huri.headSvcName" -}}
{{- printf "%s-head-svc" (include "huri.fullname" .) }}
{{- end }}
