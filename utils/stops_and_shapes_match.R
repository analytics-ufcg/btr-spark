library(dplyr)
library(readr)
library(geosphere)
library(ggplot2)

directory <- "/local/orion/bigsea/bigsea_data/curitiba_gtfs/"

stop.times <- read_csv(paste0(directory, "stop_times.txt"))
stops <- read_csv(paste0(directory, "stops.txt"))
trips <- read_csv(paste0(directory, "trips.txt"))
routes <- read_csv(paste0(directory, "routes.txt"))
shapes <- read_csv(paste0(directory, "shapes.txt"))

joined.data <- stop.times %>%
  left_join(stops, by = "stop_id") %>%
    distinct(trip_id, stop_lat, stop_lon, stop_id, stop_sequence ) %>% 
  left_join(trips, by = "trip_id") %>% 
    distinct(stop_lat, stop_lon, route_id, shape_id, stop_id, stop_sequence) %>% 
  left_join(routes, by = "route_id") %>% 
    distinct(stop_lat, stop_lon, route_id, shape_id, route_short_name, stop_id, stop_sequence) %>% 
  left_join(shapes, by = "shape_id")

joined.data <- joined.data %>% 
  rowwise() %>% 
  mutate(
    distance.stop.shape = distm (c(stop_lon, stop_lat), c(shape_pt_lon, shape_pt_lat), fun = distHaversine)
  )

stops.matched.2.shape <- joined.data %>% 
  group_by(stop_id, shape_id, route_short_name) %>% 
  summarise(distance.stop.shape = min(distance.stop.shape)) %>% 
  mutate(is.stop = TRUE)

joined.data.checking.stops <- joined.data %>% 
  left_join(stops.matched.2.shape, by=c("stop_id", "shape_id", "route_short_name", "distance.stop.shape")) %>% 
  mutate(is.stop = ifelse(is.na(is.stop),FALSE, TRUE))

stops.matched.shape.sequence <- joined.data.checking.stops %>% 
  filter(is.stop) 

write.csv(stops.matched.shape.sequence, "/local/orion/bigsea/stops-matched-shape-sequence.csv", sep = ",", row.names = FALSE)

# Validation

stops.matched.shape.sequence.2 <- stops.matched.shape.sequence %>% 
  arrange(shape_id, shape_pt_sequence) %>% 
  mutate(
    stop.dif.prev = ifelse(shape_id != lag(shape_id), 1, stop_sequence - lag(stop_sequence)),
    shape.dif.prev = ifelse(shape_id != lag(shape_id), 1, ifelse(shape_pt_sequence < lag(shape_pt_sequence), -1, 1))
  )

stops.matched.shape.sequence.2 %>% select(stop.dif.prev) %>% table()
stops.matched.shape.sequence.2 %>% select(shape.dif.prev) %>% table()
  
stops.matched.shape.sequence.3 <- stops.matched.shape.sequence.2 %>% 
  select(stop_id, shape_id, route_short_name, shape_pt_lat, shape_pt_lon, shape_pt_sequence)
  
# Joining BULMA output to stops and shape match

write.bulma.join <- function(bulma.input.filepath, bulma.output.filepath, stops.matched.seq) {
  bulma.output <- read_csv(bulma.input.filepath, na = c("", "NA", "-")) %>% 
    filter(TRIP_PROBLEM == 0)
  
  bulma.output.joined <- bulma.output %>% 
    left_join(stops.matched.seq, by = c(
      "SHAPE_SEQ" = "shape_pt_sequence",
      "SHAPE_ID" = "shape_id"
    )
    )
  
  bulma.output.joined <- bulma.output.joined %>% 
    select(-route_short_name, -shape_pt_lat, -shape_pt_lon) %>% 
    rename(STOP_ID = stop_id)
  
  write.csv(bulma.output.joined, bulma.output.filepath, row.names = FALSE, na = "-")
}

write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_01_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_01_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_02_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_02_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_03_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_03_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_04_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_04_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_05_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_05_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_06_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_06_veiculos.csv", stops.matched.shape.sequence.3)
write.bulma.join("/local/orion/bigsea/dados_bulma/output/2017_02_07_veiculos.csv","/local/orion/bigsea/dados_bulma_join_stops/2017_02_07_veiculos.csv", stops.matched.shape.sequence.3)


bulma.output <- read_csv("/local/orion/bigsea/dados_bulma/output/2017_02_01_veiculos.csv", na = c("", "NA", "-")) %>% 
  filter(TRIP_PROBLEM == 0)

bulma.output.joined <- bulma.output %>% 
  left_join(stops.matched.shape.sequence.3, by = c(
    "SHAPE_SEQ" = "shape_pt_sequence",
    "SHAPE_ID" = "shape_id"
    )
  )

bulma.output.joined %>% filter(!is.na(stop_id)) %>% count()

count.matched.stops <- bulma.output.joined %>% 
  group_by(SHAPE_ID) %>% 
  summarise(stops.number = sum(!is.na(stop_id)),
            total = n()) %>% 
  mutate(stops.by.total = stops.number / total)
  
ggplot(count.matched.stops, aes(x = "stops.by.total", y = stops.by.total)) +
  geom_violin() +
  geom_boxplot(width = 0.2)

(bulma.output.joined %>% filter(SHAPE_ID == 1713))$SHAPE_SEQ %in% (shapes %>% filter(shape_id == 1713))$shape_pt_sequence %>% table()
(bulma.output.joined %>% filter(SHAPE_ID == 1715))$SHAPE_SEQ %in% (shapes %>% filter(shape_id == 1715))$shape_pt_sequence %>% table()
(bulma.output.joined %>% filter(SHAPE_ID == 1716))$SHAPE_SEQ %in% (shapes %>% filter(shape_id == 1716))$shape_pt_sequence %>% table()
(bulma.output.joined %>% filter(SHAPE_ID == 1708))$SHAPE_SEQ %in% (shapes %>% filter(shape_id == 1708))$shape_pt_sequence %>% table()
  
  