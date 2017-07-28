library(dplyr)
library(readr)
library(geosphere)

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
  arrange(shape_id, stop_sequence) %>% 
  mutate(
    stop.dif.prev = ifelse(shape_dist_traveled == 0, 1, stop_sequence - lag(stop_sequence)),
    shape.dif.prev = ifelse(shape_dist_traveled == 0, 1, ifelse(shape_dist_traveled < lag(shape_dist_traveled), shape_dist_traveled - lag(shape_dist_traveled), 1))
  )

stops.matched.shape.sequence.2 %>% distinct(stop.dif.prev) %>% nrow()
stops.matched.shape.sequence.2 %>% distinct(shape.dif.prev) %>% nrow()
