# Libraries
library(ncdf4)
library(ggplot2)
library(reshape2)
library(caret)
library(dplyr)
library(Metrics)
library(randomForest)
library(ranger)
library(lubridate)
library(tidyr)
library(nnet)
library(viridis)
library(zoo)


# SST data
extract_sst_time_series <- function(file_path, varname = "sst_anomaly") {
  nc_data <- nc_open(file_path)
  sst_data <- ncvar_get(nc_data, varname)
  time_vals <- ncvar_get(nc_data, "time")
  daily_means <- apply(sst_data, 3, function(x) mean(x, na.rm = TRUE))
  dates <- as.POSIXct(time_vals, origin = "1970-01-01", tz = "UTC")
  nc_close(nc_data)
  data.frame(Date = as.Date(dates), SST_Anomaly = daily_means)
}


# ONI index
oni_trimonth <- data.frame(
  Year = 2018:2024,
  DJF = c(-0.9, 0.7, 0.5, -1.0, -1.0, -0.7, 1.8),
  JFM = c(-0.9, 0.7, 0.5, -0.9, -0.9, -0.4, 1.5),
  FMA = c(-0.7, 0.7, 0.4, -0.8, -1.0, -0.1, 1.1),
  MAM = c(-0.5, 0.7, 0.2, -0.7, -1.1, 0.2, 0.7),
  AMJ = c(-0.2, 0.5, -0.1, -0.5, -1.0, 0.5, 0.4),
  MJJ = c( 0.0, 0.5, -0.3, -0.4, -0.9, 0.8, 0.2),
  JJA = c( 0.1, 0.3, -0.4, -0.4, -0.8, 1.1, 0.0),
  JAS = c( 0.2, 0.1, -0.6, -0.5, -0.9, 1.3, -0.1),
  ASO = c( 0.5, 0.2, -0.9, -0.7, -1.0, 1.6, -0.2),
  SON = c( 0.8, 0.3, -1.2, -0.8, -1.0, 1.8, -0.3),
  OND = c( 0.9, 0.5, -1.3, -1.0, -0.9, 1.9, -0.4),
  NDJ = c( 0.8, 0.5, -1.2, -1.0, -0.8, 2.0, -0.5)
)

oni_long <- oni_trimonth %>%
  pivot_longer(cols = -Year, names_to = "Season", values_to = "ONI")

season_to_month <- c(
  "DJF" = 1, "JFM" = 2, "FMA" = 3, "MAM" = 4,
  "AMJ" = 5, "MJJ" = 6, "JJA" = 7, "JAS" = 8,
  "ASO" = 9, "SON" = 10, "OND" = 11, "NDJ" = 12
)

expand_oni_daily <- function(oni_long) {
  all_days <- lapply(seq_len(nrow(oni_long)), function(i) {
    yr <- oni_long$Year[i]
    seas <- oni_long$Season[i]
    val <- oni_long$ONI[i]
    
    mm <- season_to_month[seas]
    start_date <- as.Date(paste0(yr, "-", sprintf("%02d", mm), "-01"))
    end_date <- as.Date(paste0(yr, "-", sprintf("%02d", mm), "-", 
                               format(seq(start_date, by = "1 month", length.out = 2)[2] - 1, "%d")))
    
    data.frame(Date = seq(start_date, end_date, by = "day"), ONI = val)
  })
  do.call(rbind, all_days)
}


# Visualisation
extract_time_series <- function(file_path) {
  nc_data <- nc_open(file_path)
  sst_anomalies <- ncvar_get(nc_data, "sst_anomaly")
  time_vals <- ncvar_get(nc_data, "time")
  daily_means <- apply(sst_anomalies, 3, function(x) mean(x, na.rm = TRUE))
  dates <- as.POSIXct(time_vals, origin = "1970-01-01", tz = "UTC")
  nc_close(nc_data)
  data.frame(Date = as.Date(dates), SST_Anomaly = daily_means)
}

prepare_sst_data_all <- function(file_path) {
  nc_data <- nc_open(file_path)
  sst_anomalies <- ncvar_get(nc_data, "sst_anomaly")
  lon <- ncvar_get(nc_data, "longitude")
  lat <- ncvar_get(nc_data, "latitude")
  time_vals <- ncvar_get(nc_data, "time")
  nc_close(nc_data)
  
  dates <- as.POSIXct(time_vals, origin = "1970-01-01", tz = "UTC")
  dims <- dim(sst_anomalies)
  grid <- expand.grid(lon_index = 1:dims[1],
                      lat_index = 1:dims[2],
                      time_index = 1:dims[3])
  grid$SST_Anomaly <- sst_anomalies[cbind(grid$lon_index, grid$lat_index, grid$time_index)]
  grid$Longitude <- lon[grid$lon_index]
  grid$Latitude  <- lat[grid$lat_index]
  grid$Date      <- as.Date(dates[grid$time_index])
  
  df <- grid %>% select(Date, Longitude, Latitude, SST_Anomaly) %>% filter(!is.na(SST_Anomaly))
  return(df)
}

# Data preparation
setwd("C:/Users/urasd/Downloads/Data Mining/SSTA data")
sst_2018_2020 <- extract_sst_time_series("AdriaticSea_2018-2020.nc")
sst_2021_2023 <- extract_sst_time_series("AdriaticSea_3years.nc")
sst_2024      <- extract_sst_time_series("AdriaticSea_2024.nc")
sst_data <- bind_rows(sst_2018_2020, sst_2021_2023, sst_2024)

oni_daily <- expand_oni_daily(oni_long)
merged <- merge(sst_data, oni_daily, by = "Date", all.x = TRUE)
merged <- merged[complete.cases(merged$SST_Anomaly, merged$ONI), ]

train_data <- merged %>% filter(Date < as.Date("2024-01-01"))
test_data  <- merged %>% filter(Date >= as.Date("2024-01-01"))

train_data <- train_data %>%
  mutate(DayOfYear = as.numeric(format(Date, "%j")),
         DayOfYear_sin = sin(2 * pi * DayOfYear / 365),
         DayOfYear_cos = cos(2 * pi * DayOfYear / 365))
test_data <- test_data %>%
  mutate(DayOfYear = as.numeric(format(Date, "%j")),
         DayOfYear_sin = sin(2 * pi * DayOfYear / 365),
         DayOfYear_cos = cos(2 * pi * DayOfYear / 365))


# Lag features
create_lags <- function(df, lag.max = 7) {
  for(i in 1:lag.max) {
    df[[paste0("lag", i)]] <- dplyr::lag(df$SST_Anomaly, i)
  }
  df <- df[complete.cases(df), ]
  return(df)
}

train_data_lag <- create_lags(train_data, 7)

# RF modelling
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
predictors_base <- c(paste0("lag", 1:7), "ONI", "DayOfYear_sin", "DayOfYear_cos")
formula_base <- as.formula(paste("SST_Anomaly ~", paste(predictors_base, collapse = " + ")))

tunegrid <- expand.grid(
  .mtry = c(2, 3, 5, 7, 9),
  .splitrule = "variance",
  .min.node.size = c(5, 10, 20)
)

model_rf_base <- train(
  formula_base,
  data = train_data_lag,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tunegrid,
  importance = "impurity"
)

cat("Best Baseline Model parameters:\n")
print(model_rf_base$bestTune)

# Residual RF model
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
predictors_base <- c(paste0("lag", 1:7), "ONI", "DayOfYear_sin", "DayOfYear_cos")
formula_base <- as.formula(paste("SST_Anomaly ~", paste(predictors_base, collapse = " + ")))

tunegrid <- expand.grid(
  .mtry = c(2, 3, 5, 7, 9),
  .splitrule = "variance",
  .min.node.size = c(5, 10, 20)
)

model_rf_base <- train(
  formula_base,
  data = train_data_lag,
  method = "ranger",
  trControl = train_control,
  tuneGrid = tunegrid,
  importance = "impurity"
)

cat("Best Baseline Model parameters:\n")
print(model_rf_base$bestTune)



# 2024 forecast
last_known <- tail(train_data_lag, 7) %>% 
  select(Date, SST_Anomaly, ONI, DayOfYear, DayOfYear_sin, DayOfYear_cos)

val_dates <- test_data$Date
val_preds <- data.frame(Date = val_dates, 
                        Predicted_base = NA_real_,
                        Predicted_final = NA_real_)

for (i in seq_along(val_dates)) {
  current_day <- val_dates[i]
  new_row <- data.frame(Date = current_day)
  for (l in 1:7) {
    new_row[[paste0("lag", l)]] <- tail(last_known$SST_Anomaly, l)[1]
  }
  oni_current <- test_data %>% filter(Date == current_day) %>% pull(ONI)
  new_row$ONI <- if(length(oni_current) > 0) oni_current[1] else tail(last_known$ONI, 1)
  new_row$DayOfYear <- as.numeric(format(current_day, "%j"))
  new_row$DayOfYear_sin <- sin(2 * pi * new_row$DayOfYear / 365)
  new_row$DayOfYear_cos <- cos(2 * pi * new_row$DayOfYear / 365)
  
  pred_base <- predict(model_rf_base, newdata = new_row)
  val_preds$Predicted_base[i] <- pred_base
  
  new_row2 <- new_row
  pred_resid <- predict(model_rf_resid, newdata = new_row2)
  final_forecast <- pred_base + pred_resid
  val_preds$Predicted_final[i] <- final_forecast
  
  new_day_row <- data.frame(
    Date = current_day,
    SST_Anomaly = final_forecast,
    ONI = new_row$ONI,
    DayOfYear = new_row$DayOfYear,
    DayOfYear_sin = new_row$DayOfYear_sin,
    DayOfYear_cos = new_row$DayOfYear_cos
  )
  last_known <- rbind(last_known, new_day_row)
  if(nrow(last_known) > 7) {
    last_known <- tail(last_known, 7)
  }
}


# Plots
final_compare <- merge(val_preds, test_data, by = "Date", all.x = TRUE)

mae_base <- mae(final_compare$SST_Anomaly, final_compare$Predicted_base)
rmse_base <- rmse(final_compare$SST_Anomaly, final_compare$Predicted_base)
cat("Baseline MAE:", mae_base, "\n")
cat("Baseline RMSE:", rmse_base, "\n")

mae_final <- mae(final_compare$SST_Anomaly, final_compare$Predicted_final)
rmse_final <- rmse(final_compare$SST_Anomaly, final_compare$Predicted_final)
cat("Two-Stage Final MAE:", mae_final, "\n")
cat("Two-Stage Final RMSE:", rmse_final, "\n")

# Time series plot for 2024
ggplot(final_compare, aes(x = Date)) +
  geom_line(aes(y = SST_Anomaly), color = "red", size = 1, linetype = "dashed") +
  geom_line(aes(y = Predicted_base), color = "blue", size = 1) +
  geom_line(aes(y = Predicted_final), color = "green", size = 1) +
  labs(title = "Baseline (Blue) vs. Residual-Corrected (Green) vs. Actual (Red)",
       x = "Date", y = "SST Anomaly (°C)") +
  theme_minimal()


# Neural Networks
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
predictors_base <- c(paste0("lag", 1:7), "ONI", "DayOfYear_sin", "DayOfYear_cos")
formula_base <- as.formula(paste("SST_Anomaly ~", paste(predictors_base, collapse = " + ")))

model_nn_base <- train(
  formula_base,
  data = train_data_lag,
  method = "nnet",
  trControl = train_control,
  tuneLength = 5,
  linout = TRUE,
  trace = FALSE
)

cat("Best Baseline NN Model parameters:\n")
print(model_nn_base$bestTune)


# Residual model
train_data_lag$Predicted_base <- predict(model_nn_base, newdata = train_data_lag)
train_data_lag$Residual <- train_data_lag$SST_Anomaly - train_data_lag$Predicted_base

getSeasonGroup <- function(date) {
  m <- as.numeric(format(date, "%m"))
  if(m %in% c(12, 1, 2)) return("Winter")
  else if(m %in% c(3, 4, 5)) return("Spring")
  else if(m %in% c(6, 7, 8)) return("Summer")
  else return("Autumn")
}
train_data_lag$SeasonGroup <- sapply(train_data_lag$Date, getSeasonGroup)

predictors_resid <- c(paste0("lag", 1:7), "ONI", "DayOfYear_sin", "DayOfYear_cos")
formula_resid <- as.formula(paste("Residual ~", paste(predictors_resid, collapse = " + ")))

model_nn_resid <- train(
  formula_resid,
  data = train_data_lag,
  method = "nnet",
  trControl = train_control,
  tuneLength = 5,
  linout = TRUE,
  trace = FALSE
)

cat("Best Residual NN Model parameters:\n")
print(model_nn_resid$bestTune)


# 2024 forecast
last_known <- tail(train_data_lag, 7) %>% 
  select(Date, SST_Anomaly, ONI, DayOfYear, DayOfYear_sin, DayOfYear_cos)

val_dates <- test_data$Date
val_preds <- data.frame(Date = val_dates, 
                        Predicted_base = NA_real_,
                        Predicted_final = NA_real_)

for (i in seq_along(val_dates)) {
  current_day <- val_dates[i]
  new_row <- data.frame(Date = current_day)
  for (l in 1:7) {
    new_row[[paste0("lag", l)]] <- tail(last_known$SST_Anomaly, l)[1]
  }
  oni_current <- test_data %>% filter(Date == current_day) %>% pull(ONI)
  new_row$ONI <- if(length(oni_current) > 0) oni_current[1] else tail(last_known$ONI, 1)
  new_row$DayOfYear <- as.numeric(format(current_day, "%j"))
  new_row$DayOfYear_sin <- sin(2 * pi * new_row$DayOfYear / 365)
  new_row$DayOfYear_cos <- cos(2 * pi * new_row$DayOfYear / 365)
  
  pred_base <- predict(model_nn_base, newdata = new_row)
  val_preds$Predicted_base[i] <- pred_base
  
  new_row2 <- new_row
  pred_resid <- predict(model_nn_resid, newdata = new_row2)
  final_forecast <- pred_base + pred_resid
  val_preds$Predicted_final[i] <- final_forecast
  
  new_day_row <- data.frame(
    Date = current_day,
    SST_Anomaly = final_forecast,
    ONI = new_row$ONI,
    DayOfYear = new_row$DayOfYear,
    DayOfYear_sin = new_row$DayOfYear_sin,
    DayOfYear_cos = new_row$DayOfYear_cos
  )
  last_known <- rbind(last_known, new_day_row)
  if(nrow(last_known) > 7) {
    last_known <- tail(last_known, 7)
  }
}


# Plots
final_compare <- merge(val_preds, test_data, by = "Date", all.x = TRUE)

mae_base <- mae(final_compare$SST_Anomaly, final_compare$Predicted_base)
rmse_base <- rmse(final_compare$SST_Anomaly, final_compare$Predicted_base)
cat("Baseline MAE:", mae_base, "\n")
cat("Baseline RMSE:", rmse_base, "\n")

mae_final <- mae(final_compare$SST_Anomaly, final_compare$Predicted_final)
rmse_final <- rmse(final_compare$SST_Anomaly, final_compare$Predicted_final)
cat("Two-Stage Final MAE:", mae_final, "\n")
cat("Two-Stage Final RMSE:", rmse_final, "\n")

ggplot(final_compare, aes(x = Date)) +
  geom_line(aes(y = SST_Anomaly), color = "red", size = 1, linetype = "dashed") +
  geom_line(aes(y = Predicted_base), color = "blue", size = 1) +
  geom_line(aes(y = Predicted_final), color = "green", size = 1) +
  labs(title = "Baseline (Blue) vs. Two-Stage NN (Green) vs. Actual (Red)",
       x = "Date", y = "SST Anomaly (°C)") +
  theme_minimal()


# Error visualisation
mae_df <- data.frame(
  Month = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"),
  RF_Residual = c(0.0839, 0.2030, 0.4680, 1.2000, 0.4280, 0.6320, 1.4400, 1.6700, 1.0700, 0.4360, 0.8200, 0.4260),
  NN_Residual = c(0.324, 0.562, 0.657, 1.010, 0.278, 0.236, 0.936, 1.020, 0.998, 0.536, 0.493, 0.165)
)

mae_residual_long <- mae_df %>%
  pivot_longer(cols = c("RF_Residual", "NN_Residual"),
               names_to = "Model", values_to = "MAE")

mae_residual_long$Month <- factor(mae_residual_long$Month, levels = c("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"))

ggplot(mae_residual_long, aes(x = Month, y = MAE, group = Model, color = Model)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Monthly MAE Trend: RF Residual vs. NN Residual",
       x = "Month", y = "Mean Absolute Error") +
  theme_minimal()


# Daily SST anomalies timeseries
ts_2018_2020 <- extract_time_series("AdriaticSea_2018-2020.nc")
ts_2021_2023 <- extract_time_series("AdriaticSea_3years.nc")
ts_2024      <- extract_time_series("AdriaticSea_2024.nc")

ts_all <- bind_rows(ts_2018_2020, ts_2021_2023, ts_2024) %>% arrange(Date)

ggplot(ts_all, aes(x = Date, y = SST_Anomaly)) +
  geom_line(color = "black") +
  labs(title = "Daily SST Anomalies (2018–2024)",
       x = "Date", y = "SST Anomaly (°C)") +
  theme_minimal()


# Heatmaps
data_2018_2020 <- prepare_sst_data_all("AdriaticSea_2018-2020.nc")
data_2021_2023 <- prepare_sst_data_all("AdriaticSea_3years.nc")
data_2024      <- prepare_sst_data_all("AdriaticSea_2024.nc")

combined_data <- bind_rows(data_2018_2020, data_2021_2023, data_2024) %>%
  mutate(Year = format(Date, "%Y"))

mean_by_year <- combined_data %>%
  group_by(Year, Longitude, Latitude) %>%
  summarise(Mean_SST = mean(SST_Anomaly, na.rm = TRUE), .groups = "drop")

ggplot(mean_by_year, aes(x = Longitude, y = Latitude, fill = Mean_SST)) +
  geom_tile() +
  facet_wrap(~ Year, ncol = 3) +
  scale_fill_viridis_c(option = "C", limits = c(-1, 3), na.value = "grey50",
                       guide = guide_colorbar(barwidth = 0.5, barheight = 4)) +
  labs(title = "Mean SST Anomaly Heatmaps (2018–2024)",
       x = "Longitude", y = "Latitude", fill = "Mean SST Anomaly (°C)") +
  coord_fixed(1.3) +
  theme_minimal() +
  theme(plot.title = element_text(size = 10),
        legend.text = element_text(size = 8),
        legend.title = element_text(size = 8),
        legend.position = c(0.95, 0.05),
        legend.justification = c(1, 0))


# ONI SST plots with LOESS
merged_long <- merged %>%
  pivot_longer(cols = c("SST_Anomaly", "ONI"), names_to = "Variable", values_to = "Value")
sst_df <- merged_long %>%
  filter(Variable == "SST_Anomaly") %>%
  arrange(Date) %>%
  mutate(Smoothed = zoo::rollmean(Value, k = 7, fill = NA, align = "center"))

oni_df <- merged_long %>%
  filter(Variable == "ONI") %>%
  arrange(Date)


ggplot(merged_long, aes(x = Date, y = Value, color = Variable)) +
  geom_line(alpha = 0.3) +
  geom_smooth(method = "loess", se = FALSE, size = 1) +
  labs(title = "Daily SST Anomalies and ONI (Smoothed with LOESS)",
       x = "Date", y = "Value (°C or ONI Index)", color = "Variable") +
  theme_minimal()
