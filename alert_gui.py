import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import json
import requests
import webbrowser
from datetime import datetime
import folium
import os
import tempfile
from tkinter import font
import math
import time

class HospitalNavigationGUI:
    def __init__(self, master, dms_system=None):
        self.master = master
        self.dms_system = dms_system
        self.master.title("🚗 Smart Driver Monitoring System - Hospital Navigation")
        self.master.geometry("1400x900")
        self.master.configure(bg='#1e1e1e')
        
        # Azure Maps configuration (replace with your key)
        self.azure_maps_key = "azureKeyNhiHai"
        
        # Current location (default to a central location)
        self.current_location = {"lat": 21.1458, "lng": 79.0882}  # Nagpur, Maharashtra
        
        # Emergency contacts
        self.emergency_contacts = [
            {"name": "Emergency", "number": "108"},
            {"name": "Police", "number": "100"},
            {"name": "Fire", "number": "101"},
            {"name": "Family Contact", "number": "+91-9876543210"}
        ]
        
        # Fallback hospitals (Nagpur area)
        self.fallback_hospitals = [
            {
                "name": "All India Institute of Medical Sciences (AIIMS) Nagpur",
                "address": "MIHAN, Nagpur, Maharashtra 441108",
                "phone": "+91-712-2969200",
                "lat": 21.0905, "lng": 79.0506,
                "distance": 8.2, "specialties": ["Emergency", "Trauma", "Cardiology"]
            },
            {
                "name": "Orange City Hospital",
                "address": "Plot No. 21, Shrikrishna Nagar, Nagpur, Maharashtra 440015",
                "phone": "+91-712-6666900",
                "lat": 21.1059, "lng": 79.0618,
                "distance": 6.5, "specialties": ["Emergency", "ICU", "Surgery"]
            },
            {
                "name": "Kingsway Hospital",
                "address": "Kingsway, Nagpur, Maharashtra 440001",
                "phone": "+91-712-2561329",
                "lat": 21.1461, "lng": 79.0882,
                "distance": 2.1, "specialties": ["Emergency", "General Medicine"]
            },
            {
                "name": "Care Hospital",
                "address": "3, Wardha Road, Farmland, Ramdaspeth, Nagpur, Maharashtra 440012",
                "phone": "+91-712-6700700",
                "lat": 21.1302, "lng": 79.0431,
                "distance": 4.3, "specialties": ["Emergency", "Cardiology", "Neurology"]
            },
            {
                "name": "Wockhardt Hospital",
                "address": "1643, North Ambazari Road, Nagpur, Maharashtra 440033",
                "phone": "+91-712-4444444",
                "lat": 21.1617, "lng": 79.0615,
                "distance": 3.8, "specialties": ["Emergency", "Orthopedics", "Oncology"]
            }
        ]
        
        self.monitoring_active = False
        self.video_thread = None
        self.emergency_triggered = False
        self.session_start_time = time.time()
        
        self.setup_ui()
        self.create_map()
        
    def setup_ui(self):
        """Setup the main user interface"""
        # Create main container
        main_frame = tk.Frame(self.master, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        self.create_control_panel(main_frame)
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left side - Video feed and status
        left_frame = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.create_video_panel(left_frame)
        self.create_status_panel(left_frame)
        
        # Right side - Hospital navigation
        right_frame = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        right_frame.configure(width=450)
        
        self.create_navigation_panel(right_frame)
        
    def create_control_panel(self, parent):
        """Create the top control panel"""
        control_frame = tk.Frame(parent, bg='#333333', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = tk.Label(control_frame, text="🚗 Smart Driver Monitoring System", 
                              font=('Arial', 18, 'bold'), fg='#00ff00', bg='#333333')
        title_label.pack(pady=10)
        
        # Control buttons
        button_frame = tk.Frame(control_frame, bg='#333333')
        button_frame.pack(pady=(0, 10))
        
        # Start/Stop monitoring
        self.monitor_btn = tk.Button(button_frame, text="▶️ Start Monitoring", 
                                   command=self.toggle_monitoring,
                                   font=('Arial', 12, 'bold'), 
                                   bg='#00aa00', fg='white', width=15)
        self.monitor_btn.pack(side=tk.LEFT, padx=5)
        
        # Emergency button
        emergency_btn = tk.Button(button_frame, text="🚨 EMERGENCY", 
                                command=self.trigger_emergency,
                                font=('Arial', 12, 'bold'), 
                                bg='#ff0000', fg='white', width=15)
        emergency_btn.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        settings_btn = tk.Button(button_frame, text="⚙️ Settings", 
                               command=self.open_settings,
                               font=('Arial', 12), 
                               bg='#666666', fg='white', width=12)
        settings_btn.pack(side=tk.LEFT, padx=5)
        
        # Save log button
        save_btn = tk.Button(button_frame, text="💾 Save Log", 
                           command=self.save_session_log,
                           font=('Arial', 12), 
                           bg='#0066cc', fg='white', width=12)
        save_btn.pack(side=tk.LEFT, padx=5)
        
    def create_video_panel(self, parent):
        """Create video feed panel"""
        video_frame = tk.LabelFrame(parent, text="📹 Live Video Feed", 
                                   font=('Arial', 12, 'bold'), 
                                   fg='#00ff00', bg='#2d2d2d')
        video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.video_label = tk.Label(video_frame, bg='black', 
                                   text="📷 Camera feed will appear here\nClick 'Start Monitoring' to begin",
                                   font=('Arial', 14), fg='white')
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
    def create_status_panel(self, parent):
        """Create driver status panel"""
        status_frame = tk.LabelFrame(parent, text="📊 Driver Status", 
                                    font=('Arial', 12, 'bold'), 
                                    fg='#00ff00', bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Status display
        self.status_text = tk.Text(status_frame, height=8, bg='#1a1a1a', fg='#00ff00',
                                  font=('Courier', 10), state=tk.DISABLED)
        self.status_text.pack(fill=tk.X, padx=10, pady=10)
        
        # Initialize status
        self.update_status_display("System initialized - Ready to monitor")
        
    def create_navigation_panel(self, parent):
        """Create hospital navigation panel"""
        nav_frame = tk.LabelFrame(parent, text="🏥 Emergency Navigation", 
                                 font=('Arial', 12, 'bold'), 
                                 fg='#ff6600', bg='#2d2d2d')
        nav_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current location
        location_frame = tk.Frame(nav_frame, bg='#2d2d2d')
        location_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(location_frame, text="📍 Current Location:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W)
        
        self.location_label = tk.Label(location_frame, 
                                      text=f"Lat: {self.current_location['lat']:.4f}, Lng: {self.current_location['lng']:.4f}",
                                      font=('Arial', 9), fg='#cccccc', bg='#2d2d2d')
        self.location_label.pack(anchor=tk.W)
        
        # Update location button
        update_loc_btn = tk.Button(location_frame, text="📍 Update Location", 
                                  command=self.update_location,
                                  font=('Arial', 9), bg='#0066cc', fg='white')
        update_loc_btn.pack(anchor=tk.W, pady=(5, 0))
        
        # Hospital list
        hospital_frame = tk.Frame(nav_frame, bg='#2d2d2d')
        hospital_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        
        tk.Label(hospital_frame, text="🏥 Nearby Hospitals:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W)
        
        # Hospital listbox with scrollbar
        list_frame = tk.Frame(hospital_frame, bg='#2d2d2d')
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.hospital_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                          bg='#1a1a1a', fg='white', 
                                          selectbackground='#0066cc',
                                          font=('Arial', 9))
        self.hospital_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.hospital_listbox.yview)
        
        # Populate hospitals
        self.populate_hospital_list()
        
        # Navigation buttons
        nav_btn_frame = tk.Frame(nav_frame, bg='#2d2d2d')
        nav_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        navigate_btn = tk.Button(nav_btn_frame, text="🧭 Navigate to Selected", 
                               command=self.navigate_to_hospital,
                               font=('Arial', 10, 'bold'), 
                               bg='#00aa00', fg='white')
        navigate_btn.pack(fill=tk.X, pady=2)
        
        call_btn = tk.Button(nav_btn_frame, text="📞 Call Hospital", 
                           command=self.call_hospital,
                           font=('Arial', 10), 
                           bg='#ff6600', fg='white')
        call_btn.pack(fill=tk.X, pady=2)
        
        map_btn = tk.Button(nav_btn_frame, text="🗺️ View Map", 
                          command=self.show_map,
                          font=('Arial', 10), 
                          bg='#6600cc', fg='white')
        map_btn.pack(fill=tk.X, pady=2)
        
        # Emergency contacts
        self.create_emergency_contacts(nav_frame)
        
    def create_emergency_contacts(self, parent):
        """Create emergency contacts section"""
        contacts_frame = tk.LabelFrame(parent, text="☎️ Emergency Contacts", 
                                      font=('Arial', 10, 'bold'), 
                                      fg='#ff0000', bg='#2d2d2d')
        contacts_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
        
        for contact in self.emergency_contacts:
            contact_btn = tk.Button(contacts_frame, 
                                   text=f"{contact['name']}: {contact['number']}", 
                                   command=lambda num=contact['number']: self.make_call(num),
                                   font=('Arial', 9), 
                                   bg='#cc0000', fg='white',
                                   anchor=tk.W)
            contact_btn.pack(fill=tk.X, padx=5, pady=2)
    
    def populate_hospital_list(self):
        """Populate the hospital list"""
        self.hospital_listbox.delete(0, tk.END)
        for hospital in self.fallback_hospitals:
            display_text = f"{hospital['name']} ({hospital['distance']}km)"
            self.hospital_listbox.insert(tk.END, display_text)
    
    def toggle_monitoring(self):
        """Start or stop monitoring"""
        if not self.monitoring_active:
            self.start_monitoring()
        else:
            self.stop_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring system"""
        self.monitoring_active = True
        self.monitor_btn.config(text="⏹️ Stop Monitoring", bg='#cc0000')
        self.update_status_display("🔴 Monitoring started - Driver surveillance active")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.video_monitoring_loop, daemon=True)
        self.video_thread.start()
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.monitoring_active = False
        self.monitor_btn.config(text="▶️ Start Monitoring", bg='#00aa00')
        self.update_status_display("⏹️ Monitoring stopped")
        
        # Clear video display
        self.video_label.config(image='', text="📷 Camera feed stopped\nClick 'Start Monitoring' to resume")
    
    def video_monitoring_loop(self):
        """Main video monitoring loop"""
        if not self.dms_system:
            return
            
        try:
            while self.monitoring_active:
                ret, frame = self.dms_system.cap.read()
                if not ret:
                    break
                
                # Process frame
                state, annotated_frame = self.dms_system.process_frame(frame)
                
                # Update GUI
                self.master.after(0, lambda: self.update_video_display(annotated_frame))
                self.master.after(0, lambda: self.update_driver_status(state))
                
                # Check for emergency
                if state.consciousness_level in ["UNCONSCIOUS", "SEVERELY_DROWSY", "MICROSLEEP"]:
                    self.master.after(0, lambda: self.handle_emergency_state(state))
                
        except Exception as e:
            self.master.after(0, lambda: self.update_status_display(f"❌ Monitoring error: {str(e)}"))
    
    def update_video_display(self, frame):
        """Update the video display"""
        if frame is not None:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize to fit display
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)
            self.video_label.config(image=photo, text="")
            self.video_label.image = photo  # Keep a reference
    
    def update_driver_status(self, state):
        """Update driver status display"""
        status_info = f"""
🚗 Driver Status: {state.consciousness_level}
🎯 Confidence: {state.confidence:.2f}
👁️ Eye Aspect Ratio: {state.eye_aspect_ratio:.3f}
👄 Mouth Aspect Ratio: {state.mouth_aspect_ratio:.3f}
💫 Blink Rate: {state.blink_rate:.1f}/min
🎭 Head Pose: P:{state.head_pose[0]:.1f}° Y:{state.head_pose[1]:.1f}° R:{state.head_pose[2]:.1f}°
⚠️ Yawn Detected: {'Yes' if state.yawn_detected else 'No'}
😴 Microsleep: {'Yes' if state.microsleep_detected else 'No'}
⏰ Last Update: {state.timestamp.strftime('%H:%M:%S')}
        """
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(1.0, status_info.strip())
        self.status_text.config(state=tk.DISABLED)
    
    def handle_emergency_state(self, state):
        """Handle emergency state"""
        if not self.emergency_triggered:
            self.emergency_triggered = True
            self.trigger_emergency()
            
            # Auto-select nearest hospital
            if self.hospital_listbox.size() > 0:
                self.hospital_listbox.selection_set(0)  # Select first (nearest) hospital
    
    def trigger_emergency(self):
        """Trigger emergency protocol"""
        self.update_status_display("🚨 EMERGENCY TRIGGERED - Initiating emergency protocols")
        
        # Show emergency dialog
        response = messagebox.askyesno(
            "🚨 EMERGENCY ALERT", 
            "Driver emergency detected!\n\nDriver appears to be unconscious or severely drowsy.\n\nWould you like to navigate to the nearest hospital immediately?",
            icon='warning'
        )
        
        if response:
            # Auto-navigate to nearest hospital
            if self.hospital_listbox.size() > 0:
                self.hospital_listbox.selection_set(0)
                self.navigate_to_hospital()
        
        # Reset emergency flag after a delay
        self.master.after(30000, lambda: setattr(self, 'emergency_triggered', False))  # 30 seconds
    
    def navigate_to_hospital(self):
        """Navigate to selected hospital"""
        selection = self.hospital_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a hospital first!")
            return
        
        hospital = self.fallback_hospitals[selection[0]]
        self.update_status_display(f"🧭 Navigating to {hospital['name']}")
        
        # Try Azure Maps first, fallback to Google Maps
        if self.azure_maps_key and self.azure_maps_key != "YOUR_AZURE_MAPS_KEY_HERE":
            self.navigate_with_azure_maps(hospital)
        else:
            self.navigate_with_google_maps(hospital)
    
    def navigate_with_azure_maps(self, hospital):
        """Navigate using Azure Maps"""
        try:
            # Create Azure Maps route URL
            start = f"{self.current_location['lat']},{self.current_location['lng']}"
            end = f"{hospital['lat']},{hospital['lng']}"
            
            url = f"https://www.bing.com/maps?rtp=pos.{start}_name.Current%20Location~pos.{end}_name.{hospital['name']}&mode=D"
            webbrowser.open(url)
            
            messagebox.showinfo("Navigation", f"Opening route to {hospital['name']} in your default browser.")
            
        except Exception as e:
            self.update_status_display(f"❌ Azure Maps error: {str(e)}")
            self.navigate_with_google_maps(hospital)
    
    def navigate_with_google_maps(self, hospital):
        """Navigate using Google Maps as fallback"""
        try:
            # Create Google Maps route URL
            start = f"{self.current_location['lat']},{self.current_location['lng']}"
            end = f"{hospital['lat']},{hospital['lng']}"
            
            url = f"https://www.google.com/maps/dir/{start}/{end}"
            webbrowser.open(url)
            
            messagebox.showinfo("Navigation", f"Opening route to {hospital['name']} in Google Maps.")
            
        except Exception as e:
            self.update_status_display(f"❌ Navigation error: {str(e)}")
            messagebox.showerror("Error", f"Failed to open navigation: {str(e)}")
    
    def call_hospital(self):
        """Simulate calling hospital"""
        selection = self.hospital_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a hospital first!")
            return
        
        hospital = self.fallback_hospitals[selection[0]]
        
        # On a real mobile device, you would use the phone dialer
        message = f"Calling {hospital['name']}\nPhone: {hospital['phone']}\n\nOn a mobile device, this would automatically dial the number."
        messagebox.showinfo("Calling Hospital", message)
        
        self.update_status_display(f"📞 Called {hospital['name']}: {hospital['phone']}")
    
    def make_call(self, number):
        """Simulate making emergency call"""
        message = f"Calling emergency number: {number}\n\nOn a mobile device, this would automatically dial the number."
        messagebox.showinfo("Emergency Call", message)
        self.update_status_display(f"📞 Emergency call to: {number}")
    
    def show_map(self):
        """Show interactive map"""
        try:
            self.create_and_show_map()
        except Exception as e:
            messagebox.showerror("Map Error", f"Failed to create map: {str(e)}")
    
    def create_map(self):
        """Create the base map"""
        self.map_data = {
            'center': self.current_location,
            'hospitals': self.fallback_hospitals
        }
    
    def create_and_show_map(self):
        """Create and display interactive map"""
        # Create map centered on current location
        m = folium.Map(
            location=[self.current_location['lat'], self.current_location['lng']],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add current location marker
        folium.Marker(
            [self.current_location['lat'], self.current_location['lng']],
            popup="📍 Your Current Location",
            icon=folium.Icon(color='blue', icon='user')
        ).add_to(m)
        
        # Add hospital markers
        for hospital in self.fallback_hospitals:
            popup_text = f"""
            <b>{hospital['name']}</b><br>
            📍 {hospital['address']}<br>
            📞 {hospital['phone']}<br>
            🚗 {hospital['distance']} km away<br>
            🏥 Specialties: {', '.join(hospital['specialties'])}
            """
            
            folium.Marker(
                [hospital['lat'], hospital['lng']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color='red', icon='plus')
            ).add_to(m)
        
        # Save map to temporary file and open
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        m.save(temp_file.name)
        webbrowser.open(f'file://{temp_file.name}')
        
        self.update_status_display("🗺️ Interactive map opened in browser")
    
    def update_location(self):
        """Update current location (simulate GPS)"""
        # In a real application, you would get GPS coordinates
        # For demo, we'll just show a dialog to manually enter coordinates
        location_dialog = LocationUpdateDialog(self.master, self.current_location)
        if location_dialog.result:
            self.current_location = location_dialog.result
            self.location_label.config(
                text=f"Lat: {self.current_location['lat']:.4f}, Lng: {self.current_location['lng']:.4f}"
            )
            self.update_status_display(f"📍 Location updated: {self.current_location['lat']:.4f}, {self.current_location['lng']:.4f}")
            
            # Recalculate distances to hospitals
            self.calculate_hospital_distances()
    
    def calculate_hospital_distances(self):
        """Calculate distances to hospitals from current location"""
        for hospital in self.fallback_hospitals:
            distance = self.calculate_distance(
                self.current_location['lat'], self.current_location['lng'],
                hospital['lat'], hospital['lng']
            )
            hospital['distance'] = round(distance, 1)
        
        # Sort hospitals by distance
        self.fallback_hospitals.sort(key=lambda x: x['distance'])
        
        # Update hospital list display
        self.populate_hospital_list()
    
    def calculate_distance(self, lat1, lng1, lat2, lng2):
        """Calculate distance between two coordinates using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlng/2) * math.sin(dlng/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance
    
    def open_settings(self):
        """Open settings dialog"""
        settings_dialog = SettingsDialog(self.master, self)
    
    def save_session_log(self):
        """Save session log"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"driver_monitoring_session_{timestamp}.json"
            
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'session_duration': time.time() - getattr(self, 'session_start_time', time.time()),
                'current_location': self.current_location,
                'hospitals': self.fallback_hospitals,
                'emergency_contacts': self.emergency_contacts,
                'monitoring_active': self.monitoring_active,
                'emergency_triggered': self.emergency_triggered
            }
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialname=filename
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(log_data, f, indent=2)
                messagebox.showinfo("Success", f"Session log saved to {filename}")
                self.update_status_display(f"💾 Session log saved to {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {str(e)}")
    
    def update_status_display(self, message):
        """Update status display with timestamp"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        status_message = f"[{timestamp}] {message}"
        
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"\n{status_message}")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)


class LocationUpdateDialog:
    def __init__(self, parent, current_location):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("📍 Update Location")
        self.dialog.geometry("400x250")
        self.dialog.configure(bg='#2d2d2d')
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Location input frame
        input_frame = tk.Frame(self.dialog, bg='#2d2d2d')
        input_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        tk.Label(input_frame, text="📍 Update Current Location", 
                font=('Arial', 14, 'bold'), fg='white', bg='#2d2d2d').pack(pady=(0, 20))
        
        # Latitude
        tk.Label(input_frame, text="Latitude:", 
                font=('Arial', 10), fg='white', bg='#2d2d2d').pack(anchor=tk.W)
        self.lat_entry = tk.Entry(input_frame, font=('Arial', 10))
        self.lat_entry.insert(0, str(current_location['lat']))
        self.lat_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Longitude
        tk.Label(input_frame, text="Longitude:", 
                font=('Arial', 10), fg='white', bg='#2d2d2d').pack(anchor=tk.W)
        self.lng_entry = tk.Entry(input_frame, font=('Arial', 10))
        self.lng_entry.insert(0, str(current_location['lng']))
        self.lng_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons
        button_frame = tk.Frame(input_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X)
        
        tk.Button(button_frame, text="Update", command=self.update_location,
                 bg='#00aa00', fg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy,
                 bg='#666666', fg='white', font=('Arial', 10)).pack(side=tk.LEFT)
        
        # Auto-location button
        tk.Button(button_frame, text="📍 Auto-detect", command=self.auto_detect_location,
                 bg='#0066cc', fg='white', font=('Arial', 10)).pack(side=tk.RIGHT)
    
    def update_location(self):
        """Update location with entered coordinates"""
        try:
            lat = float(self.lat_entry.get())
            lng = float(self.lng_entry.get())
            
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                self.result = {"lat": lat, "lng": lng}
                self.dialog.destroy()
            else:
                messagebox.showerror("Invalid Input", "Please enter valid coordinates:\nLatitude: -90 to 90\nLongitude: -180 to 180")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric coordinates")
    
    def auto_detect_location(self):
        """Simulate auto-detection of location"""
        # In a real app, this would use GPS/geolocation API
        messagebox.showinfo("Auto-detect", "GPS auto-detection would be implemented here.\nFor demo, using Nagpur coordinates.")
        self.lat_entry.delete(0, tk.END)
        self.lat_entry.insert(0, "21.1458")
        self.lng_entry.delete(0, tk.END)
        self.lng_entry.insert(0, "79.0882")


class SettingsDialog:
    def __init__(self, parent, main_app):
        self.main_app = main_app
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("⚙️ Settings")
        self.dialog.geometry("500x600")
        self.dialog.configure(bg='#2d2d2d')
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_settings_ui()
    
    def create_settings_ui(self):
        """Create settings interface"""
        # Main container
        main_frame = tk.Frame(self.dialog, bg='#2d2d2d')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        tk.Label(main_frame, text="⚙️ System Settings", 
                font=('Arial', 16, 'bold'), fg='white', bg='#2d2d2d').pack(pady=(0, 20))
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # General settings tab
        general_frame = tk.Frame(notebook, bg='#2d2d2d')
        notebook.add(general_frame, text="General")
        self.create_general_settings(general_frame)
        
        # Emergency settings tab
        emergency_frame = tk.Frame(notebook, bg='#2d2d2d')
        notebook.add(emergency_frame, text="Emergency")
        self.create_emergency_settings(emergency_frame)
        
        # Monitoring settings tab
        monitoring_frame = tk.Frame(notebook, bg='#2d2d2d')
        notebook.add(monitoring_frame, text="Monitoring")
        self.create_monitoring_settings(monitoring_frame)
        
        # Buttons
        button_frame = tk.Frame(main_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        tk.Button(button_frame, text="Save Settings", command=self.save_settings,
                 bg='#00aa00', fg='white', font=('Arial', 12, 'bold')).pack(side=tk.LEFT)
        
        tk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults,
                 bg='#ff6600', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Button(button_frame, text="Close", command=self.dialog.destroy,
                 bg='#666666', fg='white', font=('Arial', 12)).pack(side=tk.RIGHT)
    
    def create_general_settings(self, parent):
        """Create general settings"""
        # Azure Maps API Key
        tk.Label(parent, text="Azure Maps API Key:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(10, 5))
        
        self.api_key_entry = tk.Entry(parent, font=('Arial', 10), width=50)
        self.api_key_entry.insert(0, self.main_app.azure_maps_key)
        self.api_key_entry.pack(anchor=tk.W, pady=(0, 10))
        
        # Auto-start monitoring
        self.auto_start_var = tk.BooleanVar()
        tk.Checkbutton(parent, text="Auto-start monitoring on launch", 
                      variable=self.auto_start_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=5)
        
        # Auto-save logs
        self.auto_save_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Auto-save session logs", 
                      variable=self.auto_save_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=5)
        
        # Sound alerts
        self.sound_alerts_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Enable sound alerts", 
                      variable=self.sound_alerts_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=5)
    
    def create_emergency_settings(self, parent):
        """Create emergency settings"""
        # Emergency response time
        tk.Label(parent, text="Emergency Response Time (seconds):", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(10, 5))
        
        self.response_time_var = tk.IntVar(value=10)
        response_scale = tk.Scale(parent, from_=5, to=30, orient=tk.HORIZONTAL,
                                 variable=self.response_time_var,
                                 bg='#2d2d2d', fg='white', highlightbackground='#2d2d2d')
        response_scale.pack(anchor=tk.W, pady=(0, 10))
        
        # Auto-call emergency
        self.auto_call_var = tk.BooleanVar()
        tk.Checkbutton(parent, text="Auto-call emergency services", 
                      variable=self.auto_call_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=5)
        
        # Emergency contacts management
        tk.Label(parent, text="Emergency Contacts:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(20, 5))
        
        contacts_frame = tk.Frame(parent, bg='#2d2d2d')
        contacts_frame.pack(fill=tk.X, pady=5)
        
        self.contacts_listbox = tk.Listbox(contacts_frame, height=4, bg='#1a1a1a', fg='white')
        self.contacts_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Populate contacts
        for contact in self.main_app.emergency_contacts:
            self.contacts_listbox.insert(tk.END, f"{contact['name']}: {contact['number']}")
        
        contact_buttons = tk.Frame(contacts_frame, bg='#2d2d2d')
        contact_buttons.pack(side=tk.RIGHT, padx=(10, 0))
        
        tk.Button(contact_buttons, text="Add", bg='#00aa00', fg='white', width=8).pack(pady=2)
        tk.Button(contact_buttons, text="Edit", bg='#0066cc', fg='white', width=8).pack(pady=2)
        tk.Button(contact_buttons, text="Remove", bg='#cc0000', fg='white', width=8).pack(pady=2)
    
    def create_monitoring_settings(self, parent):
        """Create monitoring settings"""
        # Detection sensitivity
        tk.Label(parent, text="Detection Sensitivity:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(10, 5))
        
        self.sensitivity_var = tk.DoubleVar(value=0.7)
        sensitivity_scale = tk.Scale(parent, from_=0.1, to=1.0, resolution=0.1,
                                   orient=tk.HORIZONTAL, variable=self.sensitivity_var,
                                   bg='#2d2d2d', fg='white', highlightbackground='#2d2d2d')
        sensitivity_scale.pack(anchor=tk.W, pady=(0, 10))
        
        # Frame rate
        tk.Label(parent, text="Video Frame Rate (FPS):", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(10, 5))
        
        self.fps_var = tk.IntVar(value=30)
        fps_scale = tk.Scale(parent, from_=10, to=60, orient=tk.HORIZONTAL,
                           variable=self.fps_var,
                           bg='#2d2d2d', fg='white', highlightbackground='#2d2d2d')
        fps_scale.pack(anchor=tk.W, pady=(0, 10))
        
        # Detection features
        tk.Label(parent, text="Detection Features:", 
                font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor=tk.W, pady=(20, 5))
        
        self.drowsiness_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Drowsiness Detection", 
                      variable=self.drowsiness_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=2)
        
        self.yawn_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Yawn Detection", 
                      variable=self.yawn_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=2)
        
        self.head_pose_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Head Pose Analysis", 
                      variable=self.head_pose_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=2)
        
        self.microsleep_var = tk.BooleanVar(value=True)
        tk.Checkbutton(parent, text="Microsleep Detection", 
                      variable=self.microsleep_var,
                      fg='white', bg='#2d2d2d', selectcolor='#2d2d2d').pack(anchor=tk.W, pady=2)
    
    def save_settings(self):
        """Save all settings"""
        try:
            # Update main app settings
            self.main_app.azure_maps_key = self.api_key_entry.get()
            
            # Save settings to file
            settings = {
                'azure_maps_key': self.api_key_entry.get(),
                'auto_start': self.auto_start_var.get(),
                'auto_save': self.auto_save_var.get(),
                'sound_alerts': self.sound_alerts_var.get(),
                'response_time': self.response_time_var.get(),
                'auto_call': self.auto_call_var.get(),
                'sensitivity': self.sensitivity_var.get(),
                'fps': self.fps_var.get(),
                'drowsiness_detection': self.drowsiness_var.get(),
                'yawn_detection': self.yawn_var.get(),
                'head_pose_analysis': self.head_pose_var.get(),
                'microsleep_detection': self.microsleep_var.get()
            }
            
            # Save to file
            with open('dms_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            messagebox.showinfo("Success", "Settings saved successfully!")
            self.main_app.update_status_display("⚙️ Settings saved")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def reset_defaults(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            # Reset all variables to defaults
            self.api_key_entry.delete(0, tk.END)
            self.api_key_entry.insert(0, "YOUR_AZURE_MAPS_KEY_HERE")
            self.auto_start_var.set(False)
            self.auto_save_var.set(True)
            self.sound_alerts_var.set(True)
            self.response_time_var.set(10)
            self.auto_call_var.set(False)
            self.sensitivity_var.set(0.7)
            self.fps_var.set(30)
            self.drowsiness_var.set(True)
            self.yawn_var.set(True)
            self.head_pose_var.set(True)
            self.microsleep_var.set(True)


def main():
    """Main function to run the hospital navigation GUI"""
    root = tk.Tk()
    
    app = HospitalNavigationGUI(root)
    
    # Load settings if available
    try:
        with open('dms_settings.json', 'r') as f:
            settings = json.load(f)
            app.azure_maps_key = settings.get('azure_maps_key', app.azure_maps_key)
            app.update_status_display("⚙️ Settings loaded from file")
    except FileNotFoundError:
        app.update_status_display("⚙️ Using default settings")
    
    # Start the GUI
    root.mainloop()


if __name__ == "__main__":
    main()