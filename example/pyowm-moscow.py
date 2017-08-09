import pyowm

API_key = '0b4ee4cd202f86472987b438b565e83c'

owm = pyowm.OWM(API_key, language='en', version='2.5')

# Will it be sunny tomorrow at this time in Milan (Italy) ?
forecast = owm.daily_forecast('Milan,it')
tomorrow = pyowm.timeutils.tomorrow()
print 'Tomorrow will be sunny (Milan,it):', forecast.will_be_sunny_at(tomorrow)  # Always True in Italy, right? ;-)
print

# Search for current weather in London (UK)
observation = owm.weather_at_place('Moscow,ru')
w = observation.get_weather()
print 'Weather (Moscow,ru):'

# Weather details
print '\tWind:', w.get_wind()                            # {'speed': 4.6, 'deg': 330}
print '\tHumidity, %:', w.get_humidity()                 # 87
print '\tTemperature, C:', w.get_temperature('celsius')  # {'temp_max': 10.5, 'temp': 9.7, 'temp_min': 9.0}