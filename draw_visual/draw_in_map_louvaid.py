import folium
from geopy.geocoders import Nominatim
from louvain_method_copy import *
from IPython.display import display
from IPython.display import IFrame

# Tạo bản đồ
m = folium.Map(location=[51.5074, 0.1278], zoom_start=5)
# Đọc file
labels = []

with open('dataset/label.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        label = line.strip()
        labels.append(label)

# Tạo đối tượng geolocator
geolocator = Nominatim(user_agent="dsafssfcxvsdfsdvsd")


true_partition, frame = louvain_algorithm(adj_matrix,7)
print(true_partition)

colors = ['blue', 'darkpurple', 'black', 'cadetblue', 'lightblue', 'darkgreen', 'gray', 'green', 'orange', 'purple', 'lightgreen', 'darkblue', 'white', 'red', 'darkred', 'pink', 'lightgray', 'lightred', 'beige']
print(len(true_partition),len(colors))
# Tìm kiếm tọa độ của các điểm và đánh dấu trên bản đồ
for i,cluster in enumerate(true_partition):
    if i>=len(colors):
        break
    for point in cluster:       
        location = geolocator.geocode(labels[point] + ", Europe")
        if location is not None:
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=point,
                icon=folium.Icon(color=colors[i])
            ).add_to(m)

# Hiển thị bản đồ

# Lưu bản đồ vào một file HTML
m.save('map.html')
