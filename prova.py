import googlemaps
from datetime import datetime

# Imposta la tua API key (non pubblicarla su repository pubblici!)
API_KEY = "AIzaSyBJND9ewKJetkmc_zkUoWjTEYMmPExLZsM"

# Inizializza il client di googlemaps
gmaps = googlemaps.Client(key=API_KEY)

# --- Esempio 1: Geocoding ---
# Geocodifica un indirizzo per ottenere le coordinate GPS
address = "1600 Amphitheatre Parkway, Mountain View, CA"
geocode_result = gmaps.geocode(address)

print("Risultato del geocoding per", address)
print(geocode_result)

# --- Esempio 2: Calcolo della Distanza ---
# Calcola il tempo di percorrenza e la distanza tra due località in modalità driving
origin = "San Francisco, CA"
destination = "Los Angeles, CA"
distance_result = gmaps.distance_matrix(origin, destination, mode="driving", departure_time=datetime.now())

print("\nRisultato della distance matrix da", origin, "a", destination)
print(distance_result)
