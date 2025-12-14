"""
Synthetic Data Generator for Airport Passenger Flow Prediction
Generates realistic passenger check-in times, flight schedules, and historical traffic data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

class AirportDataGenerator:
    """Generate synthetic airport passenger flow data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Airport configuration
        self.terminals = ['T1', 'T2', 'T3']
        self.security_gates = ['SG1', 'SG2', 'SG3', 'SG4', 'SG5']
        self.check_in_counters = [f'C{i}' for i in range(1, 21)]
        self.baggage_carousels = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6']
        
        # Airlines
        self.airlines = ['Delta', 'United', 'American', 'Southwest', 'JetBlue', 
                        'Alaska', 'Spirit', 'Frontier', 'Emirates', 'British Airways']
        
        # Flight destinations
        self.destinations = ['LAX', 'JFK', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 
                           'MIA', 'LHR', 'DXB', 'SIN', 'NRT', 'CDG', 'FRA']
        
    def generate_flight_schedule(self, start_date, num_days=30, flights_per_day=150):
        """Generate synthetic flight schedule"""
        flights = []
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            for _ in range(flights_per_day):
                # Generate departure time with realistic distribution
                # More flights in morning (6-10 AM) and evening (4-8 PM)
                hour_weights = [0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 2.5, 3.0, 3.5, 3.0,
                              2.5, 2.0, 2.0, 2.0, 2.5, 3.0, 3.5, 3.5, 3.0, 2.5,
                              2.0, 1.5, 1.0, 0.5]
                hour = random.choices(range(24), weights=hour_weights)[0]
                minute = random.randint(0, 59)
                
                departure_time = current_date.replace(hour=hour, minute=minute)
                
                # Flight details
                flight = {
                    'flight_id': f'{random.choice(self.airlines)[:2].upper()}{random.randint(100, 9999)}',
                    'airline': random.choice(self.airlines),
                    'destination': random.choice(self.destinations),
                    'terminal': random.choice(self.terminals),
                    'departure_time': departure_time,
                    'gate': f'G{random.randint(1, 50)}',
                    'aircraft_capacity': random.choice([150, 180, 200, 250, 300, 350]),
                    'booking_rate': random.uniform(0.65, 0.98),  # Percentage of seats booked
                    'is_international': random.random() > 0.7,
                    'status': random.choice(['On Time', 'On Time', 'On Time', 'Delayed', 'On Time'])
                }
                flight['expected_passengers'] = int(flight['aircraft_capacity'] * flight['booking_rate'])
                
                flights.append(flight)
        
        return pd.DataFrame(flights)
    
    def generate_passenger_checkins(self, flight_schedule, checkin_rate=0.85):
        """Generate passenger check-in events based on flight schedule"""
        checkins = []
        
        for _, flight in flight_schedule.iterrows():
            num_passengers = int(flight['expected_passengers'] * checkin_rate)
            departure = flight['departure_time']
            
            for _ in range(num_passengers):
                # Check-in time distribution: most passengers arrive 2-3 hours before
                if flight['is_international']:
                    hours_before = np.random.normal(3, 0.75)  # Earlier for international
                else:
                    hours_before = np.random.normal(2, 0.5)
                
                hours_before = max(0.5, min(4, hours_before))  # Clamp between 0.5 and 4 hours
                checkin_time = departure - timedelta(hours=hours_before)
                
                checkin = {
                    'passenger_id': f'PAX{random.randint(100000, 999999)}',
                    'flight_id': flight['flight_id'],
                    'checkin_time': checkin_time,
                    'terminal': flight['terminal'],
                    'counter': random.choice(self.check_in_counters),
                    'checkin_type': random.choice(['counter', 'counter', 'kiosk', 'online', 'online']),
                    'bags_checked': random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0],
                    'processing_time_sec': max(30, np.random.normal(180, 60))  # Average 3 min
                }
                checkins.append(checkin)
        
        return pd.DataFrame(checkins)
    
    def generate_security_queue_data(self, checkins_df, start_date, num_days=30):
        """Generate security checkpoint queue data"""
        security_data = []
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            for hour in range(24):
                for minute in range(0, 60, 15):  # Every 15 minutes
                    timestamp = current_date.replace(hour=hour, minute=minute)
                    
                    for gate in self.security_gates:
                        # Base queue length varies by time of day
                        base_queue = self._get_base_traffic(hour)
                        
                        # Add day of week variation
                        dow_factor = 1.3 if current_date.weekday() in [4, 6] else 1.0  # Fri, Sun busier
                        
                        # Add randomness
                        queue_length = int(base_queue * dow_factor * np.random.uniform(0.7, 1.3))
                        
                        # Wait time correlates with queue length (approx 2 min per person)
                        wait_time = queue_length * np.random.uniform(1.5, 2.5)
                        
                        security_data.append({
                            'timestamp': timestamp,
                            'security_gate': gate,
                            'queue_length': max(0, queue_length),
                            'wait_time_minutes': max(0, round(wait_time, 1)),
                            'throughput_per_hour': random.randint(100, 180),
                            'lanes_open': random.randint(2, 6),
                            'precheck_available': random.random() > 0.3
                        })
        
        return pd.DataFrame(security_data)
    
    def generate_baggage_claim_data(self, flight_schedule, start_date, num_days=30):
        """Generate baggage claim crowding data"""
        baggage_data = []
        
        # Filter for arriving flights (simulate as half the flights)
        arrivals = flight_schedule.sample(frac=0.5)
        
        for _, flight in arrivals.iterrows():
            arrival_time = flight['departure_time']  # Using departure as arrival for simulation
            
            # Baggage typically starts 15-30 min after landing
            baggage_start = arrival_time + timedelta(minutes=random.randint(15, 30))
            
            carousel = random.choice(self.baggage_carousels)
            num_bags = int(flight['expected_passengers'] * 1.5)  # Avg 1.5 bags per passenger
            
            # Generate time series data for this flight's baggage claim
            for offset_min in range(0, 45, 5):  # Track for 45 minutes
                timestamp = baggage_start + timedelta(minutes=offset_min)
                
                # Crowding peaks at 10-20 min, then decreases
                if offset_min < 10:
                    crowd_factor = offset_min / 10
                elif offset_min < 25:
                    crowd_factor = 1.0
                else:
                    crowd_factor = max(0, 1 - (offset_min - 25) / 20)
                
                passengers_waiting = int(flight['expected_passengers'] * crowd_factor * random.uniform(0.6, 0.9))
                
                baggage_data.append({
                    'timestamp': timestamp,
                    'carousel': carousel,
                    'flight_id': flight['flight_id'],
                    'passengers_waiting': passengers_waiting,
                    'bags_remaining': int(num_bags * max(0, 1 - offset_min / 40)),
                    'congestion_level': self._get_congestion_level(passengers_waiting)
                })
        
        return pd.DataFrame(baggage_data)
    
    def generate_historical_traffic(self, start_date, num_days=365):
        """Generate historical daily traffic patterns for training"""
        traffic_data = []
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            for hour in range(24):
                # Base traffic pattern
                base_traffic = self._get_base_traffic(hour)
                
                # Seasonal variation (higher in summer and holidays)
                month = current_date.month
                seasonal_factor = 1 + 0.3 * np.sin((month - 3) * np.pi / 6)  # Peak in summer
                
                # Day of week variation
                dow = current_date.weekday()
                dow_factors = [0.85, 0.9, 0.95, 1.0, 1.2, 1.15, 1.1]  # Mon-Sun
                dow_factor = dow_factors[dow]
                
                # Holiday boost
                holiday_factor = 1.0
                if current_date.month == 12 and current_date.day in range(20, 32):
                    holiday_factor = 1.5  # Christmas period
                elif current_date.month == 11 and current_date.day in range(22, 29):
                    holiday_factor = 1.4  # Thanksgiving
                elif current_date.month == 7 and current_date.day in range(1, 8):
                    holiday_factor = 1.3  # July 4th week
                
                total_passengers = int(base_traffic * seasonal_factor * dow_factor * 
                                      holiday_factor * np.random.uniform(0.85, 1.15))
                
                traffic_data.append({
                    'date': current_date,
                    'hour': hour,
                    'timestamp': current_date.replace(hour=hour),
                    'total_passengers': total_passengers,
                    'departures': int(total_passengers * 0.5),
                    'arrivals': int(total_passengers * 0.5),
                    'day_of_week': dow,
                    'month': month,
                    'is_weekend': dow >= 5,
                    'is_holiday': holiday_factor > 1.0,
                    'weather_impact': random.uniform(0.8, 1.0),  # Weather delay factor
                    'avg_security_wait': max(5, np.random.normal(15, 5)),
                    'avg_checkin_wait': max(2, np.random.normal(8, 3))
                })
        
        return pd.DataFrame(traffic_data)
    
    def _get_base_traffic(self, hour):
        """Get base traffic level for a given hour"""
        # Typical airport traffic pattern
        hourly_pattern = {
            0: 50, 1: 30, 2: 20, 3: 20, 4: 40, 5: 100,
            6: 300, 7: 500, 8: 600, 9: 550, 10: 450, 11: 400,
            12: 400, 13: 450, 14: 500, 15: 550, 16: 600, 17: 650,
            18: 600, 19: 500, 20: 400, 21: 300, 22: 200, 23: 100
        }
        return hourly_pattern.get(hour, 200)
    
    def _get_congestion_level(self, passengers):
        """Categorize congestion level"""
        if passengers < 20:
            return 'low'
        elif passengers < 50:
            return 'moderate'
        elif passengers < 80:
            return 'high'
        else:
            return 'critical'
    
    def generate_all_data(self, output_dir, start_date=None, historical_days=365, 
                         forecast_days=30, flights_per_day=150):
        """Generate all datasets and save to files"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=historical_days)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating historical traffic data...")
        historical_traffic = self.generate_historical_traffic(start_date, historical_days)
        historical_traffic.to_csv(f'{output_dir}/historical_traffic.csv', index=False)
        print(f"  - Saved {len(historical_traffic)} records to historical_traffic.csv")
        
        print("Generating flight schedule...")
        recent_start = datetime.now() - timedelta(days=forecast_days)
        flight_schedule = self.generate_flight_schedule(recent_start, forecast_days, flights_per_day)
        flight_schedule.to_csv(f'{output_dir}/flight_schedule.csv', index=False)
        print(f"  - Saved {len(flight_schedule)} flights to flight_schedule.csv")
        
        print("Generating passenger check-in data...")
        checkins = self.generate_passenger_checkins(flight_schedule)
        checkins.to_csv(f'{output_dir}/passenger_checkins.csv', index=False)
        print(f"  - Saved {len(checkins)} check-ins to passenger_checkins.csv")
        
        print("Generating security queue data...")
        security_data = self.generate_security_queue_data(checkins, recent_start, forecast_days)
        security_data.to_csv(f'{output_dir}/security_queue.csv', index=False)
        print(f"  - Saved {len(security_data)} records to security_queue.csv")
        
        print("Generating baggage claim data...")
        baggage_data = self.generate_baggage_claim_data(flight_schedule, recent_start, forecast_days)
        baggage_data.to_csv(f'{output_dir}/baggage_claim.csv', index=False)
        print(f"  - Saved {len(baggage_data)} records to baggage_claim.csv")
        
        print("\n✅ All data generated successfully!")
        
        return {
            'historical_traffic': historical_traffic,
            'flight_schedule': flight_schedule,
            'checkins': checkins,
            'security_data': security_data,
            'baggage_data': baggage_data
        }


if __name__ == '__main__':
    generator = AirportDataGenerator(seed=42)
    data = generator.generate_all_data(
        output_dir='./generated_data',
        historical_days=365,
        forecast_days=30,
        flights_per_day=150
    )
