import numpy as np
import pandas as pd
import datetime
import pprint
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import random


def visualize_dataset(data_dict, output_filename="dataset_visualization.png"):
    """
    Visualizes a dataset stored in a dictionary where each key is a date string (YYYY-MM-DD)
    and each value is a tuple of (float, float, string).

    Parameters:
    - data_dict (dict): Dictionary with 365 key-value pairs.
    - output_filename (str): Name of the output PNG file.
    """
    # Sort the data by date
    dates = sorted(data_dict.keys())
    x_values = [date if isinstance(date, datetime.date) else datetime.datetime.strptime(date, "%Y-%m-%d").date() for date in dates]
    y1_values = [data_dict[date][0] for date in dates]
    y2_values = [data_dict[date][1] for date in dates]
    phases = [data_dict[date][2] for date in dates]

    # Define colors for phases
    phase_colors = {
        'Dormancy': 'red',
        'Budbreak': 'orange',
        'Veraison': 'yellow',
        'Maturity': 'green'
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot lines with segment coloring
    prev_phase = phases[0]
    segment_start = 0
    
    for i in range(1, len(dates)):
        if phases[i] != prev_phase:
            ax.plot(x_values[segment_start:i], y1_values[segment_start:i], color=phase_colors[prev_phase], linewidth=2, label='Chilling Units' if segment_start == 0 else "")
            ax.plot(x_values[segment_start:i], y2_values[segment_start:i], color=phase_colors[prev_phase], linestyle='dashed', linewidth=2, label='Forcing Units' if segment_start == 0 else "")
            segment_start = i
            prev_phase = phases[i]
    
    # Plot the last segment
    ax.plot(x_values[segment_start:], y1_values[segment_start:], color=phase_colors[prev_phase], linewidth=2, label='Chilling Units')
    ax.plot(x_values[segment_start:], y2_values[segment_start:], color=phase_colors[prev_phase], linestyle='dashed', linewidth=2, label='Forcing Units')
    
    # Markers for phase transitions
    for i in range(1, len(dates)):
        if phases[i] != phases[i - 1]:
            ax.scatter(x_values[i], y1_values[i], color='black', marker='o', s=50, label=phases[i])
            ax.scatter(x_values[i], y2_values[i], color='black', marker='x', s=50)
    
    # Formatting x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.set_title("Dataset Visualization Over 365 Days")
    
    # Add legend
    legend_patches = [plt.Line2D([0], [0], color=color, lw=2, label=phase) for phase, color in phase_colors.items()]
    legend_patches.append(plt.Line2D([0], [0], color='black', lw=2, label='Chilling Units'))
    legend_patches.append(plt.Line2D([0], [0], color='black', linestyle='dashed', lw=2, label='Forcing Units'))
    ax.legend(handles=legend_patches, title="Phases & Units")
    
    # Save and show plot
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()


class VineyardModel:
    def __init__(self, season_length=300, GDD_threshold=150, T_base=10):
        # Simulation parameters
        self.season_length = season_length
        self.GDD_threshold = GDD_threshold  # Required GDD for bud break
        self.T_base = T_base  # Base temperature for GDD calculation

        """Initialize parameters with default values and their descriptions"""
        # Phenological development
        self.aParam = 0.005  # Curve shape parameter (unitless)
        self.cParam = 2.8  # Optimal chilling temperature (°C)
        self.ChillingReq = 50.692  # Chilling requirement (CU)
        self.db = -0.26  # Slope of forcing unit equation for budbreak (unitless)
        self.df = -0.26  # Slope of forcing unit equation for flowering (unitless)
        self.dv = -0.26  # Slope of forcing unit equation for veraison (unitless)
        self.dm = -0.26  # Slope of forcing unit equation for maturity (unitless)
        self.eb = 16.06  # Inflection point for budbreak (°C)
        self.ef = 16.06  # Inflection point for flowering (°C)
        self.ev = 16.06  # Inflection point for veraison (°C)
        self.em = 16.06  # Inflection point for maturity (°C)
        self.Col = 176.26  # Curve shape parameter (unitless)
        self.co2 = -0.015  # Curve shape parameter (unitless)
        self.LimitForcingReq = 234  # Last day of chilling effect on forcing requirement (d)
        self.FloweringReq = 24.71  # Forcing requirement for flowering (FU)
        self.VeraisonReq = 51.146  # Forcing requirement for veraison (FU)
        self.MaturityReq = None  # Forcing requirement for maturity (FU, missing default)
        self.budbreaktresh = 0
        self.veraisontresh = 300
        self.floweringtresh = 650
        self.maturitytresh = 1000
        # Leaf growth
        self.ShootNumber = 16  # Number of shoots per plant (n° of shoots)
        self.SLNI = -0.28  # Curve shape parameter of Shoot Leaf Number equation (n° of leaves d−1)
        self.LAR1 = 0.04  # Curve shape parameter of Shoot Leaf Number equation (n° of leaves d−1 °C-1)
        self.LAR2 = -0.015  # Curve shape parameter of Shoot Leaf Number equation (n° of leaves−1)
        self.SLAS = 5.39  # Curve shape parameter of Shoot Leaf Area equation (cm² n° of leaves−1)
        self.SLAE = 2.13  # Curve shape parameter of Shoot Leaf Area equation (unitless)
        self.ProportionShadedArea = 0.75  # Proportion of the area shaded by plant (unitless)
        self.PlantingDensity = 4  # Squared meters of soil per plant (m² plant−1)
        self.SLN1 = 25.9  # Coefficient of water stress equation on leaf development (unitless)
        self.SLN2 = 17.3  # Coefficient of water stress equation on leaf development (unitless)

        # Light interception and biomass accumulation
        self.CropCoeff = 0.6  # Extinction coefficient for intercepted radiation (unitless)
        self.InitialRUE = 1.001  # Initial RUE at CO2 concentration of 350 ppm (g MJ−1)
        self.qRUE550 = 0.633  # Intercept of the linear equation for CO2 550 ppm (g MJ−1)
        self.mRUE550 = 0.00105  # Coeff. of the linear equation for CO2 550 ppm (g MJ−1 ppm−1)
        self.PHO1 = 12.9  # Coefficient for water stress effect on photosynthesis (unitless)
        self.PHO2 = 14.1  # Coefficient for water stress effect on photosynthesis (unitless)
        self.qRUE700 = 0.954  # Intercept of the linear equation for CO2 700 ppm (g MJ−1)
        self.mRUE700 = 0.000468  # Coeff. of the linear equation for CO2 700 ppm (g MJ−1 ppm−1)

        # Biomass partitioning
        self.HI = 0.00443  # Slope of Fruit Biomass Index equation (d−1)
        self.FruitSetIndex = 0.8
        self.HICutOff = 0.5  # Harvest Index cut off (d−1)
        self.Rootini = 100  # Initial value of root depth (cm)
        self.RootGrowthBasic = 0.001  # Root growth rate for not limited water conditions (cm d−1)
        self.RootGrowthStress = 0.05  # Root growth rate for limited water conditions (cm d−1)
        self.FTSWLimitRoot = 0.5  # FTSW Limit for Root Growth (unitless)
        self.DayForStress = 2  # N° days after which water stress effect on root occurs (d)

        # Evapotranspiration
        self.TEC = 6.1  # Transpiration efficiency coefficient (Pa)

        # Extreme event impact
        self.Tmax = 41  # Maximum temperature for fruit set at flowering (°C)
        self.Tmin = 1  # Minimum temperature for fruit set at flowering (°C)
        self.Topt = 25  # Optimum temperature for fruit set at flowering (°C)
        self.q = 1.9  # Curve shape parameter (unitless)


        # Initial state variables
        self.day = 0
        self.D = 1  # Dormancy state (1 = dormant, 0 = active)
        self.GDD = 0  # Accumulated Growing Degree Days

        # Growth variables
        self.V_b = 0.01  # Branch volume (m³)
        self.M_b = 5  # Branch biomass (g dry weight)
        self.N_l = 0  # Number of leaves
        self.A_l = 0  # Leaf surface area (m²)
        self.M_l = 0  # Leaf biomass (g dry weight)
        self.G_p = 0  # Grape presence (0 = no, 1 = yes)
        self.V_g = 0  # Grape volume (m³)
        self.M_g = 0  # Grape biomass (g dry weight)
        self.S_g = 0  # Grape sugar content (°Brix)
        self.Maturity_g = 0  # Grape maturity stage (0-3)

        # Environmental inputs
        self.T = []
        self.I = []
        self.CO2 = []
        self.SM = []

        # Storage for tracking phenological stage
        self.budbreakreq = 0
        self.floweringreq = 0
        self.veraisonreq = 0
        self.maturityreq = 0

        # Storage for phenological dates
        self.budbreakdate = None
        self.floweringdate = None
        self.veraisondate = None
        self.maturitydate = None

         # Define base temperatures for different stages
        self.basetempbudbreak = 7  # °C
        self.basetempbloom = 9     # °C
        self.basetempveraison = 11  # °C
        self.basetempharvest = 12   # °C
        
        # Define GDD thresholds for each stage
        self.GDDbudbreakreq = 100
        self.GDDbloomreq = 300
        self.GDDveraisonreq = 650
        self.GDDharvestreq = 1000
        
        # Tracking if a stage has been reached (0 = not reached, 1 = reached)
        self.GDDbudbreak = 0
        self.GDDbloom = 0
        self.GDDveraison = 0
        self.GDDharvest = 0
    
    def light_intensity_acquisition(self, file_path):
        """ Reads light intensity from a CSV file and returns both a dictionary and a list """
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        light_intensity_dict = {}
        light_intensity_list = []
        last_valid_value = 0  # Default initial value

        for date, intensity in zip(df['Date'], df['Light Intensity (MJ/m²)']):
            try:
                intensity = float(intensity)
                last_valid_value = intensity
            except ValueError:
                print(f"For day {date.date()} there's no valid value")
                intensity = last_valid_value  # Use last valid value
            light_intensity_dict[date.date()] = intensity
            light_intensity_list.append(intensity)
        self.I = light_intensity_list
        return light_intensity_dict, light_intensity_list
    
    def soil_moisture_acquisition(self, file_path):
        """ Reads soil moisture from a CSV file and returns both a dictionary and a list """
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        soil_moisture_dict = {}
        soil_moisture_list = []
        last_valid_value = 0  # Default initial value

        for date, soil_moisture in zip(df['Date'], df['Soil Moisture (%)']):
            try:
                soil_moisture = float(soil_moisture)
                last_valid_value = soil_moisture
            except ValueError:
                print(f"For day {date.date()} there's no valid value")
                soil_moisture = last_valid_value  # Use last valid value
            soil_moisture_dict[date.date()] = soil_moisture
            soil_moisture_list.append(soil_moisture)
        self.SM = soil_moisture_list
        return soil_moisture_dict, soil_moisture_list

    def temperature_acquisition(self, file_path):
        """ Reads soil moisture from a CSV file and returns both a dictionary and a list """
        df = pd.read_csv(file_path)
        df['PragaDate'] = pd.to_datetime(df['PragaDate'])
        temperature_dict = {}
        temperature_list = []
        last_valid_value = 0  # Default initial value PragaDate,DAILY_TMIN,DAILY_TMAX

        for date, temp_min, temp_max in zip(df['PragaDate'], df['DAILY_TMIN'], df['DAILY_TMAX']):
            try:
                # Ensure values are valid (check for NaN)
                if pd.isna(temp_min) or pd.isna(temp_max):
                    raise ValueError  # Triggers fallback to last valid value
                
                temp_min = float(temp_min)
                temp_max = float(temp_max)
                average_temp = (temp_max + temp_min)/2
                last_valid_value = average_temp

            except ValueError:
            # If invalid, use the last valid value
                if last_valid_value is None:
                    print(f"Warning: No valid temperature found before {date.date()}. Setting default as None.")
                    average_temp = None
                else:
                    print(f"For day {date.date()}, no valid temperature data. Using last valid value.")
                    average_temp = last_valid_value  # Use previous valid temp

            #store values
            temperature_dict[date.date()] = average_temp
            temperature_list.append(average_temp)

        self.T = temperature_list

        return temperature_dict, temperature_list
    
        '''
        df_temp = pd.read_csv(config["temperature_file"])
        df_temp["T_avg"] = (df_temp["DAILY_TMIN"] + df_temp["DAILY_TMAX"]) / 2
        df_temp = df_temp.sort_values(by="PragaDate")
        self.T = df_temp["T_avg"].tolist()
        '''


    def co2_acquisition(self, file_path):
        """ Reads CO2 data from a CSV file, extracts only 2023 data, and fills missing values with linear interpolation. """

        # Read CSV
        df = pd.read_csv(file_path)

        # Ensure necessary columns exist
        if 'date' not in df.columns or 'value' not in df.columns:
            raise ValueError("CSV file must contain 'date' and 'value' columns.")

        # Convert 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Filter data for the year 2023
        df = df[(df['date'].dt.year == 2023)]

        # Initialize storage
        co2_dict = {}
        co2_list = []
        
        # Define the full date range for 2023
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2023, 12, 31)
        full_date_range = pd.date_range(start=start_date, end=end_date)

        # Convert DataFrame into a dictionary of known values
        co2_data = dict(zip(df['date'].dt.date, df['value']))

        # Handle missing start date (set all previous values to 400 if 2023-01-01 is missing)
        first_available_date = min(co2_data.keys()) if co2_data else None
        default_value = 400

        for current_date in full_date_range:
            current_date = current_date.date()  # Convert to native date type

            if current_date in co2_data:
                # If the date exists, use its value
                co2_value = float(co2_data[current_date])
            
            elif first_available_date and current_date < first_available_date:
                # If before first available date, set to 400
                co2_value = default_value
            
            else:
                # If missing, perform linear interpolation
                previous_date = max([d for d in co2_dict.keys() if d < current_date], default=None)
                next_date = min([d for d in co2_data.keys() if d > current_date], default=None)

                if previous_date is not None and next_date is not None:
                    # Interpolation formula: y = y1 + (y2 - y1) * ((x - x1) / (x2 - x1))
                    y1 = co2_dict[previous_date]  # Previous known value
                    y2 = co2_data[next_date]      # Next known value
                    x1 = (previous_date - start_date).days  # Convert date to day index
                    x2 = (next_date - start_date).days
                    x = (current_date - start_date).days

                    co2_value = y1 + (y2 - y1) * ((x - x1) / (x2 - x1))

                else:
                    # If there's no previous or next data, default to 400 (shouldn't happen)
                    co2_value = default_value

            # Store values
            co2_dict[current_date] = co2_value
            co2_list.append(co2_value)

        # Update class attribute
        self.CO2 = co2_list

        return co2_dict, co2_list

    '''
    def set_environmental_inputs(self, T, I, CO2, SM):
        """ Set environmental conditions for the simulation """
        self.T = T
        self.I = I
        self.CO2 = CO2
        self.SM = SM         
    '''
    def phenological_stages(self):
        """Tracks phenological development using chilling-forcing units method."""
        
        # Initialize variables
        chilling_units = 0
        forcing_units = 0
        current_stage = "Dormancy"
        tracking_data = {}

        # Initialize date tracking (starting at 2023-01-01)
        start_date = datetime.date(2023, 1, 1)
        
        chilling_complete = False  # Flag for chilling completion
        maturity_reached = False  # Flag for stopping forcing accumulation
        
        for day_index in range(self.season_length):
            # Compute current date
            current_date = start_date + datetime.timedelta(days=day_index)
            T_avg = self.T[day_index]

            # --- CHILLING ACCUMULATION ---
            if not chilling_complete:
                cu = 1 / (1 + np.exp(self.aParam * ((T_avg - self.cParam) ** 2)))
                chilling_units += cu
                
                # Stop chilling accumulation once threshold is reached
                if chilling_units >= self.ChillingReq:
                    
                    chilling_complete = True
                    forcing_units = 0  # Reset forcing units for the next phase

            # --- FORCING ACCUMULATION ---
            elif not maturity_reached:
                
                # Check phenological thresholds and update stages
                if forcing_units >= self.budbreaktresh and forcing_units < self.floweringtresh:
                    self.budbreakreq=1
                    
                    if T_avg > self.basetempbudbreak:  # Base temperature for budbreak
                        forcing_units += T_avg - self.basetempbudbreak  # GDD-style accumulation
                    self.budbreakdate = current_date
                    current_stage = "Budbreak"
                
                elif forcing_units > self.floweringtresh and forcing_units < self.veraisontresh:
                    self.floweringreq=1
                    
                    if T_avg > self.basetempbloom:  # Base temperature for budbreak
                        forcing_units += T_avg - self.basetempbloom  # GDD-style accumulation
                    self.floweringdate = current_date
                    current_stage = "Flowering"
                

                elif forcing_units > self.veraisontresh and forcing_units < self.maturitytresh:
                    self.veraisonreq=1
                    if T_avg > self.basetempveraison:  # Base temperature for budbreak
                        forcing_units += T_avg - self.basetempveraison # GDD-style accumulation
                    self.veraisondate = current_date
                    current_stage = "Veraison"

                elif forcing_units > self.maturitytresh:
                    self.maturityreq = 1
                    maturity_reached = True

            elif maturity_reached:
                if T_avg > self.basetempharvest:  # Base temperature for budbreak
                    forcing_units += T_avg - self.basetempharvest  # GDD-style accumulation
                self.maturitydatedate = current_date
                current_stage = "Maturity"

                '''
                # Check phenological thresholds and update stages
                if self.budbreakreq == 0 and forcing_units >= self.budbreaktresh:
                    self.budbreakreq = 1
                    self.budbreakdate = current_date
                    current_stage = "Budbreak"

                elif self.floweringreq == 0 and forcing_units >= self.floweringtresh:
                    self.floweringreq = 1
                    self.floweringdate = current_date
                    current_stage = "Flowering"

                elif self.veraisonreq == 0 and forcing_units >= self.veraisontresh:
                    self.veraisonreq = 1
                    self.veraisondate = current_date
                    current_stage = "Veraison"

                elif self.maturityreq == 0 and forcing_units >= self.maturitytresh:
                    self.maturityreq = 1
                    self.maturitydate = current_date
                    current_stage = "Maturity"
                    maturity_reached = True  # Stop forcing accumulation
                '''
            # Store the daily tracking data
            tracking_data[current_date] = (chilling_units, forcing_units, current_stage)

        return tracking_data
    
    def GDD_approach(self):
        """
        Compute GDD accumulation and update phenological stages in sequential order.
        
        Returns:
        - dict: Keys are day indices (1-based), values are tuples (current GDD, current stage).
        """
        
        # Initialize accumulations
        gdd_accumulation = 0
        
        # Define stage order
        stages = [
            ("Dormancy", None, None),
            ("Budbreak", "GDDbudbreak", "GDDbudbreakreq"),
            ("Bloom", "GDDbloom", "GDDbloomreq"),
            ("Veraison", "GDDveraison", "GDDveraisonreq"),
            ("Harvest", "GDDharvest", "GDDharvestreq")
        ]
        
        # Initialize current stage index
        stage_index = 0
        
        # Dictionary to store daily outputs
        phenology_timeline = {}
        
        for day in range(self.season_length):
            T_avg = self.T[day]  # Get daily average temperature
            
            # Check if still in dormancy
            if stage_index == 0:
                phenology_timeline[day + 1] = (gdd_accumulation, "Dormancy")
                continue
            
            # Get current stage and parameters
            stage_name, stage_flag, stage_threshold = stages[stage_index]
            T_base = getattr(self, f"basetemp{stage_name.lower()}")
            
            # Calculate daily GDD contribution
            if T_avg > T_base:
                gdd_accumulation += T_avg - T_base
                
            # Check if stage requirement is met
            if gdd_accumulation >= getattr(self, stage_threshold) and getattr(self, stage_flag) == 0:
                setattr(self, stage_flag, 1)  # Mark stage as reached
                stage_index += 1  # Move to next stage
                
            # Store current day record
            phenology_timeline[day + 1] = (gdd_accumulation, stages[stage_index][0])
        
        return phenology_timeline

    def update_GDD(self, T_today):
        """ Accumulate Growing Degree Days and check for bud break """
        if T_today > self.T_base:
            self.GDD += (T_today - self.T_base)
        if self.GDD >= self.GDD_threshold:
            self.D = 0  # Bud break occurs
            self.N_l = 2  # Initial leaves appear
            self.A_l = 0.02  # Initial leaf area (m²)
            self.V_b = 0.05  # Initial branch volume (m³)

    def photosynthesis(self):
        """
        Compute photosynthesis and biomass accumulation for the season.
        
        Returns:
        - dict: Keys are day indices (1-based), values are tuples (PhotoRate, VegBio, FruBio).
        - list: Total biomass values for each day.
        """
        
        shoot_leaf_number = 0  # Initialize shoot leaf number
        tot_bio = 0  # Total biomass
        biomass_list = []  # List to store daily total biomass
        phenology_timeline = {}  # Dictionary for daily biomass data
        branch_biomass = [] # List to store daily branch biomass
        leaf_biomass = []
        fruit_biomass = []
        for day in range(self.season_length):
            T_avg = self.T[day]
            I_day = self.I[day]
            CO2_day = self.CO2[day]
            
            # Update ShootLeafNumber
            shoot_leaf_number += self.SLNI + self.LAR1 * T_avg + self.LAR2 * shoot_leaf_number
            
            # Compute ShootLeafArea
            shoot_leaf_area = self.SLAS * (shoot_leaf_number ** self.SLAE)
            
            # Compute PlantLeafArea
            plant_leaf_area = shoot_leaf_area * self.ShootNumber
            
            # Compute Leaf Area Index (LAI)
            leaf_area_index = (plant_leaf_area * self.ProportionShadedArea) / 10000
            
            # Compute intercepted radiation using Beer-Lambert’s Law
            rad_intercepted = 1 - (2.718 ** (-self.CropCoeff * leaf_area_index))
            
            # Compute RUEmax based on CO2 levels
            if CO2_day <= 350:
                RUE_max = self.InitialRUE
            elif 350 < CO2_day <= 550:
                RUE_max = self.mRUE550 * CO2_day + self.qRUE550
            else:
                RUE_max = self.mRUE700 * CO2_day + self.qRUE700
            
            # Compute RUE considering temperature effect
            RUE = RUE_max * (1 - 0.0025 * (((self.T[day] * 1.16667) - 25) ** 2))

            # Compute daily photosynthesis rate
            photo_rate = I_day * rad_intercepted * self.ProportionShadedArea * RUE

            # Accumulate total biomass
            tot_bio += photo_rate
            
            # Biomass partitioning based on phenological stages
            if self.floweringreq == 1 and self.veraisonreq == 0:  # Budbreak → Pre-Flowering
                branch_bio = tot_bio * 0.7
                leaf_bio = tot_bio * 0.3
                fru_bio = 0
            elif self.veraisonreq == 1 and self.maturityreq ==0:  # Flowering → Veraison
                fru_bio = tot_bio * self.HI * self.FruitSetIndex
                branch_bio = (tot_bio - fru_bio) * 0.2
                leaf_bio = (tot_bio - fru_bio) * 0.8
                
            elif self.maturityreq == 1:  # Veraison → Harvest
                fru_bio = tot_bio * self.HI * self.FruitSetIndex
                branch_bio = (tot_bio - fru_bio) * 0.6
                leaf_bio = (tot_bio - fru_bio) * 0.4
            else: #it means we are in the budbreak
                branch_bio = tot_bio 
                leaf_bio =0
                fru_bio = 0
            
            # Store daily data
            branch_bio = abs(branch_bio)
            leaf_bio = abs(leaf_bio)
            fru_bio = abs(fru_bio)
            tot_bio = abs(tot_bio)
            phenology_timeline[day + 1] = (photo_rate, leaf_bio, branch_bio, fru_bio)
            biomass_list.append(tot_bio)
            branch_biomass.append(branch_bio)
            leaf_biomass.append(leaf_bio)
            fruit_biomass.append(fru_bio)
        
        
        return phenology_timeline, biomass_list, branch_biomass, leaf_biomass, fruit_biomass
    
    def compute_photosynthesis(self, T_today, I_today, CO2_today, SM_today):
        """ Compute daily photosynthesis rate based on environment """
        I_today = max(I_today, 0.01)  # Prevent division by zero
        P_max = (15 * I_today) / (I_today + 10)
        f_T = np.exp(-((T_today - 25) / 10) ** 2)
        f_CO2 = CO2_today / (CO2_today + 300)
        f_SM = min(1, SM_today / 30)
        P_final = P_max * f_T * f_CO2 * f_SM
        return P_final * self.A_l * 0.02  # Carbon assimilation (g C/day)
    
    def allocate_carbon(self, C_total):
        """ Allocate carbon to leaves, branches, and grapes """
        C_l = C_total * 0.6  # 60% to leaves early season
        C_b = C_total * 0.4  # 40% to branches early season

        if self.N_l > 50:  # Condition for grape formation
            self.G_p = 1

        if self.G_p:
            C_g = C_total * 0.5  # Later, 50% to grapes
            self.M_g += C_g
            self.V_g = max(0, self.V_g + C_g / 1000)  # Prevent negative volume
        
        self.M_l += C_l
        self.A_l += C_l / 50  # Leaf area increase (m²)
        self.N_l += int(C_l / 10)  # Ensure integer leaf count
        self.M_b += C_b
        self.V_b += C_b / 200  # Branch volume increase (m³)
    
    def update_phenological_stages(self):
        """ Update grape sugar accumulation and maturity """
        if self.G_p:
            self.S_g += 0.2 * np.exp(-((self.T[self.day] - 25) / 10) ** 2)
            self.S_g = min(self.S_g, 30)  # Cap sugar content
            if self.S_g >= 10:
                self.Maturity_g = 1  # Ripening
            if self.S_g >= 20:
                self.Maturity_g = 2  # Fully ripe
            if self.S_g >= 25:
                self.Maturity_g = 3  # Overripe
    
    def check_environmental_stress(self):
        """ Check for heat or drought stress impacting grape drop """
        if self.T[self.day] > 35 or self.SM[self.day] < 15:
            self.N_l = int(self.N_l * 0.98)  # Ensure integer leaf count
            self.V_g = max(0, self.V_g * 0.95)  # Prevent negative grape volume
    
    def apply_daily_updates(self, T_today, I_today, CO2_today, SM_today):
        """ Perform a single day evaluation based on user input """
        self.update_GDD(T_today)
        if self.D == 0:
            C_total = self.compute_photosynthesis(T_today, I_today, CO2_today, SM_today)
            self.allocate_carbon(C_total)
            self.update_phenological_stages()
            self.check_environmental_stress()
        
        self.day += 1  # Increment day counter
        result = {
            "day": self.day,
            "Branch Volume (m³)": self.V_b,
            "Leaf Count": self.N_l,
            "Leaf Area (m²)": self.A_l,
            "Grape Volume (m³)": self.V_g,
            "Sugar Content (°Brix)": self.S_g,
            "Maturity Stage": self.Maturity_g
        }
        pprint.pprint(result)
        return result

'''  
obj = VineyardModel()
co2_dict, co2_list = obj.co2_acquisition("co2_file.csv")
temperature_dict, temperature_list = obj.temperature_acquisition("temperature_file.csv")
soil_moisture_dict, soil_moisture_list = obj.soil_moisture_acquisition("soil_moisture_file.csv")
light_intensity_dict, light_intensity_list = obj.light_intensity_acquisition("light_intensity_file.csv")
tracking_data = obj.phenological_stages()

for key, value in tracking_data.items():
    print(f"{key}: {value}")

phenology_timeline, biomass_list, branch_biomass, leaf_biomass, fruit_biomass = obj.photosynthesis()
print(f"{branch_biomass}", end="\n")
'''
