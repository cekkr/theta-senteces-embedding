import math
import numpy as np

class Theta2:
    def __init__(self, x=None, y=None, degrees=0):
        """
        Initialize a Theta2 object either with x,y coordinates or with degrees.
        By default, creates a Theta2 at 0 degrees (1,0).
        """
        if x is not None and y is not None:
            # Initialize from coordinates
            self.x = x
            self.y = y
            self._normalize()
        else:
            # Initialize from degrees
            radians = math.radians(degrees)
            self.x = math.cos(radians)
            self.y = math.sin(radians)
    
    def _normalize(self):
        """Normalize coordinates to unit circle"""
        magnitude = math.sqrt(self.x**2 + self.y**2)
        if magnitude > 0:
            self.x = self.x / magnitude
            self.y = self.y / magnitude
    
    def get_degrees(self):
        """Get the angle in degrees"""
        return math.degrees(math.atan2(self.y, self.x)) % 360
    
    def get_radians(self):
        """Get the angle in radians"""
        return math.atan2(self.y, self.x) % (2 * math.pi)
    
    def rotate(self, degrees):
        """Rotate the angle by the specified degrees"""
        radians = math.radians(degrees)
        cos_alpha = math.cos(radians)
        sin_alpha = math.sin(radians)
        
        x_new = self.x * cos_alpha - self.y * sin_alpha
        y_new = self.x * sin_alpha + self.y * cos_alpha
        
        self.x = x_new
        self.y = y_new
        return self
    
    def __add__(self, degrees):
        """Add degrees to the angle"""
        return self.copy().rotate(degrees)
    
    def __sub__(self, degrees):
        """Subtract degrees from the angle"""
        return self.copy().rotate(-degrees)
    
    def copy(self):
        """Create a copy of this Theta2"""
        return Theta2(self.x, self.y)
    
    def compare(self, other):
        """
        Compare angles using cross product.
        Returns:
        - Positive: self is counterclockwise from other
        - Negative: self is clockwise from other
        - Near zero: angles are similar (or opposite)
        """
        cross = self.x * other.y - self.y * other.x
        return cross
    
    def is_greater_than(self, other):
        """Check if this angle is greater than another (in counterclockwise direction)"""
        return self.compare(other) > 0
    
    def angle_between(self, other):
        """Find the angle between two theta objects (in radians)"""
        dot = self.x * other.x + self.y * other.y
        det = self.x * other.y - self.y * other.x
        angle = math.atan2(det, dot)
        return angle
    
    def angle_between_degrees(self, other):
        """Find the angle between two theta objects (in degrees)"""
        return math.degrees(self.angle_between(other))
    
    def rotate_90(self):
        """Rotate by 90 degrees quickly"""
        self.x, self.y = -self.y, self.x
        return self
    
    def rotate_180(self):
        """Rotate by 180 degrees quickly"""
        self.x, self.y = -self.x, -self.y
        return self
    
    def rotate_270(self):
        """Rotate by 270 degrees quickly"""
        self.x, self.y = self.y, -self.x
        return self
    
    def __repr__(self):
        return f"Theta2(x={self.x:.6f}, y={self.y:.6f}, degrees={self.get_degrees():.2f}°)"


class IntTheta2:
    # Precomputed rotation matrices for common angles (scaled by 1000)
    # Format: (cos, sin, -sin, cos)
    ROTATIONS = {
        90: (0, 1000, -1000, 0),
        180: (-1000, 0, 0, -1000),
        270: (0, -1000, 1000, 0),
        45: (707, 707, -707, 707),
        135: (-707, 707, -707, -707),
        225: (-707, -707, 707, -707),
        315: (707, -707, 707, 707)
    }
    
    # Precomputed CORDIC angles in BAM format (scaled by 2^16)
    CORDIC_ANGLES = [
        11520,  # 45.0 degrees
        6801,   # 26.57 degrees
        3593,   # 14.04 degrees
        1824,   # 7.13 degrees
        916,    # 3.58 degrees
        458,    # 1.79 degrees
        229,    # 0.89 degrees
        115,    # 0.44 degrees
        57,     # 0.22 degrees
        29      # 0.11 degrees
    ]
    
    # CORDIC gain factor (scaled by 1024)
    CORDIC_GAIN = 607  # ~= 1/1.647 * 1000
    
    def __init__(self, x=None, y=None, degrees=0, precision=1000):
        """
        Initialize an IntTheta2 object either with x,y coordinates or with degrees.
        precision determines the scaling factor for integer representation
        """
        self.precision = precision
        
        if x is not None and y is not None:
            # Initialize from coordinates (could be float or int)
            if isinstance(x, int) and isinstance(y, int):
                self.x = x
                self.y = y
            else:
                self.x = int(x * precision)
                self.y = int(y * precision)
            self._normalize()
        else:
            # Initialize from degrees
            radians = math.radians(degrees)
            self.x = int(math.cos(radians) * precision)
            self.y = int(math.sin(radians) * precision)
    
    def _normalize(self):
        """Normalize integer coordinates to approximate unit circle"""
        # Using integer square root approximation
        magnitude_squared = self.x**2 + self.y**2
        magnitude = int(math.sqrt(magnitude_squared))
        
        if magnitude > 0:
            self.x = (self.x * self.precision) // magnitude
            self.y = (self.y * self.precision) // magnitude
    
    def fast_normalize(self):
        """
        Fast normalization using maximum norm approximation.
        Less accurate but much faster for integer math.
        """
        abs_x, abs_y = abs(self.x), abs(self.y)
        
        # Quick maximum norm approximation: max + min/2
        approximate_length = max(abs_x, abs_y) + (min(abs_x, abs_y) >> 1)
        
        if approximate_length > 0:
            self.x = (self.x * self.precision) // approximate_length
            self.y = (self.y * self.precision) // approximate_length
        
        return self
    
    def get_degrees(self):
        """Get the angle in degrees"""
        # Convert to float for atan2 calculation
        return math.degrees(math.atan2(self.y, self.x)) % 360
    
    def get_bam(self):
        """
        Get the angle in Binary Angle Measurement (BAM) format.
        Returns a value between 0-65535 representing 0-360 degrees.
        """
        degrees = self.get_degrees()
        return int((degrees * 65536) / 360) & 0xFFFF
    
    def set_from_bam(self, bam):
        """Set angle from BAM value (0-65535)"""
        degrees = (bam * 360) / 65536
        radians = math.radians(degrees)
        self.x = int(math.cos(radians) * self.precision)
        self.y = int(math.sin(radians) * self.precision)
        return self
    
    def rotate(self, degrees):
        """Rotate the angle by the specified degrees"""
        radians = math.radians(degrees)
        cos_alpha = math.cos(radians)
        sin_alpha = math.sin(radians)
        
        # Use integer arithmetic with scaling
        x_new = int(self.x * cos_alpha - self.y * sin_alpha)
        y_new = int(self.x * sin_alpha + self.y * cos_alpha)
        
        self.x = x_new
        self.y = y_new
        self._normalize()
        return self
    
    def rotate_fast(self, degrees):
        """
        Rotate by common angles using lookup table.
        If angle not in table, falls back to standard rotation.
        """
        # Check if we have a precomputed rotation
        if degrees in self.ROTATIONS:
            cos_a, sin_a, neg_sin_a, cos_a2 = self.ROTATIONS[degrees]
            
            # Apply rotation matrix with scaling
            x_new = (self.x * cos_a + self.y * neg_sin_a) // 1000
            y_new = (self.x * sin_a + self.y * cos_a2) // 1000
            
            self.x = x_new
            self.y = y_new
            self.fast_normalize()
            return self
        
        # Fall back to standard rotation
        return self.rotate(degrees)
    
    def rotate_90(self):
        """Rotate by 90 degrees quickly by swapping coordinates"""
        self.x, self.y = -self.y, self.x
        return self
    
    def rotate_180(self):
        """Rotate by 180 degrees quickly by negating both coordinates"""
        self.x, self.y = -self.x, -self.y
        return self
    
    def rotate_270(self):
        """Rotate by 270 degrees quickly by swapping coordinates"""
        self.x, self.y = self.y, -self.x
        return self
    
    def cordic_rotate(self, angle_degrees, iterations=8):
        """
        Rotate using CORDIC algorithm - very efficient for integer math.
        angle_degrees: angle in degrees to rotate by
        iterations: number of CORDIC iterations (more = higher precision)
        """
        # Convert angle to BAM format (0-65535 = 0-360 degrees)
        target_angle = int((angle_degrees % 360) * 65536 / 360)
        
        # Start with current position
        x, y = self.x, self.y
        
        # Current accumulated angle (in BAM)
        current_angle = 0
        
        # CORDIC iterations
        for i in range(min(iterations, len(self.CORDIC_ANGLES))):
            # Determine rotation direction
            if current_angle < target_angle:
                # Rotate counterclockwise
                x_new = x - (y >> i)
                y_new = y + (x >> i)
                current_angle += self.CORDIC_ANGLES[i]
            else:
                # Rotate clockwise
                x_new = x + (y >> i)
                y_new = y - (x >> i)
                current_angle -= self.CORDIC_ANGLES[i]
            
            x, y = x_new, y_new
        
        # Apply CORDIC gain compensation
        self.x = (x * self.CORDIC_GAIN) >> 10  # Divide by 1024
        self.y = (y * self.CORDIC_GAIN) >> 10
        
        self.fast_normalize()
        return self
    
    def compare(self, other):
        """
        Compare angles using cross product.
        Returns:
        - Positive: self is counterclockwise from other
        - Negative: self is clockwise from other
        - Near zero: angles are similar (or opposite)
        """
        cross = self.x * other.y - self.y * other.x
        return cross
    
    def is_greater_than(self, other):
        """Check if this angle is greater than another (in counterclockwise direction)"""
        return self.compare(other) > 0
    
    def angle_between(self, other):
        """
        Calculate the angle between two IntTheta2 objects in integer BAM units.
        Returns a value between 0-65535.
        """
        # Calculate dot and cross products
        dot = self.x * other.x + self.y * other.y
        cross = self.x * other.y - self.y * other.x
        
        # Use lookup table or approximation for atan2
        # For simplicity, we'll convert to floats here
        # In a full implementation, you'd use a fast integer atan2 approximation
        angle_radians = math.atan2(cross, dot)
        
        # Convert to BAM units (0-65535)
        bam_angle = int((angle_radians % (2 * math.pi)) * 65536 / (2 * math.pi))
        return bam_angle
    
    def angle_between_degrees(self, other):
        """Calculate the angle between two IntTheta2 objects in degrees."""
        bam_angle = self.angle_between(other)
        return (bam_angle * 360) / 65536
    
    def __add__(self, degrees):
        """Add degrees to the angle"""
        return self.copy().rotate(degrees)
    
    def __sub__(self, degrees):
        """Subtract degrees from the angle"""
        return self.copy().rotate(-degrees)
    
    def copy(self):
        """Create a copy of this IntTheta2"""
        result = IntTheta2(0, 0, precision=self.precision)
        result.x = self.x
        result.y = self.y
        return result
    
    def __repr__(self):
        return f"IntTheta2(x={self.x}, y={self.y}, degrees={self.get_degrees():.2f}°)"


class BamTheta:
    """
    Binary Angle Measurement implementation of Theta.
    Uses 16-bit integer (0-65535) to represent 0-360 degrees.
    Extremely efficient for integer operations.
    """
    def __init__(self, angle_degrees=0):
        # Convert degrees to BAM units (0-65535)
        self.angle = int((angle_degrees % 360) * 65536 / 360) & 0xFFFF
    
    def get_degrees(self):
        """Convert BAM angle to degrees"""
        return (self.angle * 360) / 65536
    
    def rotate(self, angle_degrees):
        """Rotate by specified degrees"""
        # Convert degrees to BAM units
        bam_units = int((angle_degrees % 360) * 65536 / 360)
        # Add and wrap to 16-bit
        self.angle = (self.angle + bam_units) & 0xFFFF
        return self
    
    def get_sin_cos(self, precision=1000):
        """
        Get sine and cosine of the angle as integers.
        Uses lookup table or simple approximation.
        """
        # For a real implementation, you'd use a lookup table
        # Here we'll use math functions and convert to integer
        radians = (self.angle * 2 * math.pi) / 65536
        return (int(math.sin(radians) * precision), 
                int(math.cos(radians) * precision))
    
    def __add__(self, angle_degrees):
        """Add degrees to the angle"""
        result = self.copy()
        result.rotate(angle_degrees)
        return result
    
    def __sub__(self, angle_degrees):
        """Subtract degrees from the angle"""
        result = self.copy()
        result.rotate(-angle_degrees)
        return result
    
    def compare(self, other):
        """
        Compare two BAM angles.
        Returns signed difference (positive if self > other).
        """
        diff = (self.angle - other.angle) & 0xFFFF
        # Convert to signed (-32768 to 32767)
        if diff > 32767:
            diff -= 65536
        return diff
    
    def is_greater_than(self, other):
        """Test if this angle is greater than another"""
        # Calculate shortest angular distance
        diff = self.compare(other)
        return 0 < diff < 32768
    
    def copy(self):
        """Create a copy of this BamTheta"""
        result = BamTheta()
        result.angle = self.angle
        return result
    
    def __repr__(self):
        return f"BamTheta(angle={self.angle}, degrees={self.get_degrees():.2f}°)"


def demo():
    print("=== Float Theta2 Demo ===")
    theta = Theta2()  # Default is 0 degrees
    print(f"Initial theta: {theta}")
    
    # Rotate by adding 45 degrees
    theta_45 = theta + 45
    print(f"Theta + 45°: {theta_45}")
    
    # Rotate by subtracting 30 degrees
    theta_minus_30 = theta - 30
    print(f"Theta - 30°: {theta_minus_30}")
    
    # In-place rotation
    theta.rotate(90)
    print(f"After rotating 90°: {theta}")
    
    # Create from specific coordinates
    theta_custom = Theta2(0.7071, 0.7071)  # Should be close to 45 degrees
    print(f"Custom theta from coordinates: {theta_custom}")
    
    # Test angle comparison
    print(f"Angle between 0° and 45°: {theta.angle_between_degrees(theta_45):.2f}°")
    print(f"Is 90° > 45°? {theta.is_greater_than(theta_45)}")
    
    # Test fast rotations
    theta_90 = Theta2()
    theta_90.rotate_90()
    print(f"After rotate_90(): {theta_90}")
    
    print("\n=== Integer Theta2 Demo ===")
    int_theta = IntTheta2()  # Default is 0 degrees
    print(f"Initial int theta: {int_theta}")
    
    # Rotate by adding 45 degrees
    int_theta_45 = int_theta + 45
    print(f"Int Theta + 45°: {int_theta_45}")
    
    # Fast rotation methods
    int_theta_90 = IntTheta2()
    int_theta_90.rotate_90()
    print(f"After rotate_90(): {int_theta_90}")
    
    int_theta_fast = IntTheta2()
    int_theta_fast.rotate_fast(45)
    print(f"After rotate_fast(45): {int_theta_fast}")
    
    int_theta_cordic = IntTheta2()
    int_theta_cordic.cordic_rotate(45)
    print(f"After cordic_rotate(45): {int_theta_cordic}")
    
    # Angle comparison
    print(f"Int angle between 0° and 45°: {int_theta.angle_between_degrees(int_theta_45):.2f}°")
    print(f"Is 90° > 45°? {int_theta_90.is_greater_than(int_theta_45)}")
    
    print("\n=== Binary Angle Measurement (BAM) Demo ===")
    bam_theta = BamTheta()  # 0 degrees
    print(f"Initial BAM theta: {bam_theta}")
    
    bam_theta_45 = BamTheta(45)
    print(f"BAM theta at 45°: {bam_theta_45}")
    
    # Rotation
    bam_theta.rotate(90)
    print(f"After rotating 90°: {bam_theta}")
    
    # Comparison
    print(f"Is 90° > 45°? {bam_theta.is_greater_than(bam_theta_45)}")
    
    # Get sine and cosine
    sin, cos = bam_theta.get_sin_cos()
    print(f"sin(90°) = {sin/1000}, cos(90°) = {cos/1000}")
    
    print("\n=== Rotation Accuracy Test ===")
    print("Float implementation:")
    test_theta = Theta2()
    for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        rotated = test_theta + angle
        print(f"Rotated to {angle}°: actual={rotated.get_degrees():.2f}°")
    
    print("\nInteger implementation:")
    test_int_theta = IntTheta2()
    for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        rotated = test_int_theta + angle
        print(f"Rotated to {angle}°: actual={rotated.get_degrees():.2f}°")
    
    print("\nInteger CORDIC implementation:")
    test_cordic_theta = IntTheta2()
    for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        rotated = test_cordic_theta.copy().cordic_rotate(angle)
        print(f"Rotated to {angle}°: actual={rotated.get_degrees():.2f}°")
    
    print("\nBAM implementation:")
    test_bam_theta = BamTheta()
    for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        rotated = test_bam_theta + angle
        print(f"Rotated to {angle}°: actual={rotated.get_degrees():.2f}°")

if __name__ == "__main__":
    demo()