## Ampliamos el rango de iteraciones para cubrir hasta 2012-11-15
max_iter = 1495152 #(hasta 2012-11-15T14:00:00)
model_days = 430 # Total de días cubiertos por el modelo

seasons = ["JFM","ASO"]

## Dias
# Tiempos correspondientes a los 91 días de JFM
idx_t_JFM_days = [i for i in range(model_days) if (i>=110 and i<201)]
# Tiempos correspondientes a los 92 días de ASO 2012
idx_t_ASO_days = [i for i in range(model_days) if (i>=323 and i<415)]

## Horas
# Tiempos correspondientes a los 91*24 horas de JFM
idx_t_JFM_hours = [(24*i)+hr for i in range(model_days) if (i>=110 and i<201) for hr in range(24)]
# Tiempos correspondientes a los 92*24 horas de ASO 2012
idx_t_ASO_hours = [(24*i)+hr for i in range(model_days) if (i>=323 and i<415) for hr in range(24)]


## Estructura que engloba todo
idx_t = {
    "days": {
        "JFM": idx_t_JFM_days,
        "ASO": idx_t_ASO_days,
        "JFMASO": idx_t_JFM_days+idx_t_ASO_days
    },
    "hours": {
        "JFM": idx_t_JFM_hours,
        "ASO": idx_t_ASO_hours,
        "JFMASO": idx_t_JFM_hours+idx_t_ASO_hours
    }
}
