/**
 * BamTheta implementation for WebGPU
 * 
 * Uses Binary Angle Measurement (BAM) to represent angles as 16-bit integers (0-65535)
 * for efficient GPU computation.
 */
class BamThetaWebGPU {
    constructor() {
      this.adapter = null;
      this.device = null;
      this.computeShader = null;
      this.computePipeline = null;
      this.bindGroupLayout = null;
      this.workgroupSize = 64; // Default workgroup size
      this.initialized = false;
    }
  
    /**
     * Initialize WebGPU device and create shader module
     * @returns {Promise<BamThetaWebGPU>} The initialized instance
     */
    async initialize() {
      if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
      }
  
      this.adapter = await navigator.gpu.requestAdapter({
        featureLevel: 'compatibility'
      });
      
      if (!this.adapter) {
        throw new Error("Failed to get GPU adapter.");
      }
  
      this.device = await this.adapter.requestDevice();
      
      if (!this.device) {
        throw new Error("Failed to get GPU device.");
      }
      
      // Create shader module with the BamTheta implementation
      this.computeShader = this.device.createShaderModule({
        code: `
          // Input arrays and configuration
          struct BamConfig {
            workgroupSize: u32,
            operation: u32, // 0 = add, 1 = subtract, 2 = compare, etc.
          }
          
          @group(0) @binding(0) var<uniform> config: BamConfig;
          @group(0) @binding(1) var<storage, read> input1: array<u32>; // Use u32 for 16-bit BAM values
          @group(0) @binding(2) var<storage, read> input2: array<u32>;
          @group(0) @binding(3) var<storage, write> output: array<u32>;
  
          // Simple BAM operations
          fn addBam(a: u32, b: u32) -> u32 {
            return (a + b) & 0xFFFFu; // 16-bit wrap (0-65535)
          }
  
          fn subtractBam(a: u32, b: u32) -> u32 {
            return (a - b) & 0xFFFFu; // 16-bit wrap (0-65535)
          }
          
          fn compareBam(a: u32, b: u32) -> i32 {
            var diff = (a as i32) - (b as i32);
            if (diff > 32767) {
              diff -= 65536;
            } else if (diff < -32768) {
              diff += 65536;
            }
            return diff;
          }
  
          // Get sine from BAM angle using lookup or approximation
          fn bamSin(angle: u32) -> f32 {
            let angleNormalized = f32(angle) / 65536.0;
            return sin(angleNormalized * 6.283185307179586); // 2*PI
          }
  
          // Get cosine from BAM angle using lookup or approximation
          fn bamCos(angle: u32) -> f32 {
            let angleNormalized = f32(angle) / 65536.0;
            return cos(angleNormalized * 6.283185307179586); // 2*PI
          }
  
          // Main compute shader function
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            
            // Ensure we don't go out of bounds
            if (idx >= arrayLength(&output)) {
              return;
            }
            
            // Select operation based on config
            switch(config.operation) {
              case 0u: { // Add
                output[idx] = addBam(input1[idx], input2[idx]);
                break;
              }
              case 1u: { // Subtract
                output[idx] = subtractBam(input1[idx], input2[idx]);
                break;
              }
              case 2u: { // Compare
                output[idx] = u32(compareBam(input1[idx], input2[idx]));
                break;
              }
              default: { // Default to add
                output[idx] = addBam(input1[idx], input2[idx]);
                break;
              }
            }
          }
        `
      });
      
      // Create the bind group layout
      this.bindGroupLayout = this.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'uniform' }
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'read-only-storage' }
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: 'storage' }
          }
        ]
      });
      
      // Create the compute pipeline
      const pipelineLayout = this.device.createPipelineLayout({
        bindGroupLayouts: [this.bindGroupLayout]
      });
      
      this.computePipeline = this.device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: this.computeShader,
          entryPoint: 'main',
          constants: {
            workgroupSize: this.workgroupSize
          }
        }
      });
      
      this.initialized = true;
      return this;
    }
    
    /**
     * Process two arrays of BAM angles using the specified operation
     * @param {Uint32Array} bamArray1 - First array of BAM angles
     * @param {Uint32Array} bamArray2 - Second array of BAM angles
     * @param {number} operation - Operation code (0=add, 1=subtract, 2=compare)
     * @returns {Promise<Uint32Array>} Resulting BAM angles
     */
    async processBamArrays(bamArray1, bamArray2, operation = 0) {
      if (!this.initialized) {
        await this.initialize();
      }
      
      const length = bamArray1.length;
      
      if (length !== bamArray2.length) {
        throw new Error("Input arrays must have the same length");
      }
      
      // Create config buffer
      const configBuffer = this.device.createBuffer({
        size: 8, // 2 u32 values
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint32Array(configBuffer.getMappedRange()).set([this.workgroupSize, operation]);
      configBuffer.unmap();
      
      // Create buffers for input and output data
      const input1Buffer = this.device.createBuffer({
        size: bamArray1.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint32Array(input1Buffer.getMappedRange()).set(bamArray1);
      input1Buffer.unmap();
      
      const input2Buffer = this.device.createBuffer({
        size: bamArray2.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Uint32Array(input2Buffer.getMappedRange()).set(bamArray2);
      input2Buffer.unmap();
      
      // Create buffer for the output
      const outputBuffer = this.device.createBuffer({
        size: length * 4, // u32 values (4 bytes each)
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      // Create a buffer for reading back the results
      const resultBuffer = this.device.createBuffer({
        size: length * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      
      try {
        // Create the bind group
        const bindGroup = this.device.createBindGroup({
          layout: this.bindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: configBuffer } },
            { binding: 1, resource: { buffer: input1Buffer } },
            { binding: 2, resource: { buffer: input2Buffer } },
            { binding: 3, resource: { buffer: outputBuffer } }
          ]
        });
        
        // Create command encoder and compute pass
        const commandEncoder = this.device.createCommandEncoder();
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, bindGroup);
        
        // Dispatch workgroups - calculate based on data size and workgroup size
        const workgroupsNeeded = Math.ceil(length / this.workgroupSize);
        computePass.dispatchWorkgroups(workgroupsNeeded, 1, 1);
        computePass.end();
        
        // Copy the output buffer to the result buffer for reading
        commandEncoder.copyBufferToBuffer(
          outputBuffer, 0,
          resultBuffer, 0,
          length * 4
        );
        
        // Submit the command buffer
        this.device.queue.submit([commandEncoder.finish()]);
        
        // Read back the results
        await resultBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Uint32Array(resultBuffer.getMappedRange());
        
        // Create a copy of the result data (since we need to unmap the buffer)
        const resultCopy = new Uint32Array(resultData);
        
        resultBuffer.unmap();
        
        // Clean up resources that aren't needed anymore
        setTimeout(() => {
          resultBuffer.destroy();
          input1Buffer.destroy();
          input2Buffer.destroy();
          outputBuffer.destroy();
          configBuffer.destroy();
        }, 0);
        
        return resultCopy;
      } catch (error) {
        console.error("Error during WebGPU computation:", error);
        
        // Clean up in case of error
        resultBuffer.destroy();
        input1Buffer.destroy();
        input2Buffer.destroy();
        outputBuffer.destroy();
        configBuffer.destroy();
        
        throw error;
      }
    }
    
    /**
     * Add two arrays of angles (in degrees)
     * @param {number[]} thetaArray1 - First array of angles in degrees
     * @param {number[]} thetaArray2 - Second array of angles in degrees
     * @returns {Promise<number[]>} Resulting angles in degrees
     */
    async addThetaArrays(thetaArray1, thetaArray2) {
      const length = thetaArray1.length;
      
      // Convert degrees arrays to BAM arrays
      const bamArray1 = new Uint32Array(length);
      const bamArray2 = new Uint32Array(length);
      
      for (let i = 0; i < length; i++) {
        bamArray1[i] = degreesToBam(thetaArray1[i]);
        bamArray2[i] = degreesToBam(thetaArray2[i]);
      }
      
      // Process with operation = 0 (add)
      const resultBam = await this.processBamArrays(bamArray1, bamArray2, 0);
      
      // Convert back to degrees
      const resultDegrees = new Array(length);
      for (let i = 0; i < length; i++) {
        resultDegrees[i] = bamToDegrees(resultBam[i]);
      }
      
      return resultDegrees;
    }
    
    /**
     * Subtract second array of angles from first (in degrees)
     * @param {number[]} thetaArray1 - First array of angles in degrees
     * @param {number[]} thetaArray2 - Second array of angles in degrees
     * @returns {Promise<number[]>} Resulting angles in degrees
     */
    async subtractThetaArrays(thetaArray1, thetaArray2) {
      const length = thetaArray1.length;
      
      // Convert degrees arrays to BAM arrays
      const bamArray1 = new Uint32Array(length);
      const bamArray2 = new Uint32Array(length);
      
      for (let i = 0; i < length; i++) {
        bamArray1[i] = degreesToBam(thetaArray1[i]);
        bamArray2[i] = degreesToBam(thetaArray2[i]);
      }
      
      // Process with operation = 1 (subtract)
      const resultBam = await this.processBamArrays(bamArray1, bamArray2, 1);
      
      // Convert back to degrees
      const resultDegrees = new Array(length);
      for (let i = 0; i < length; i++) {
        resultDegrees[i] = bamToDegrees(resultBam[i]);
      }
      
      return resultDegrees;
    }
    
    /**
     * Compare two arrays of angles (in degrees)
     * @param {number[]} thetaArray1 - First array of angles in degrees
     * @param {number[]} thetaArray2 - Second array of angles in degrees
     * @returns {Promise<number[]>} Signed differences between angles
     */
    async compareThetaArrays(thetaArray1, thetaArray2) {
      const length = thetaArray1.length;
      
      // Convert degrees arrays to BAM arrays
      const bamArray1 = new Uint32Array(length);
      const bamArray2 = new Uint32Array(length);
      
      for (let i = 0; i < length; i++) {
        bamArray1[i] = degreesToBam(thetaArray1[i]);
        bamArray2[i] = degreesToBam(thetaArray2[i]);
      }
      
      // Process with operation = 2 (compare)
      const resultBam = await this.processBamArrays(bamArray1, bamArray2, 2);
      
      // Convert to signed differences (don't convert to degrees for comparison)
      const resultDiffs = new Array(length);
      for (let i = 0; i < length; i++) {
        let diff = resultBam[i];
        // Convert to signed (-32768 to 32767)
        if (diff > 32767) {
          diff -= 65536;
        }
        resultDiffs[i] = diff;
      }
      
      return resultDiffs;
    }
    
    /**
     * Set workgroup size (must be power of 2)
     * @param {number} size - Workgroup size (default 64)
     */
    setWorkgroupSize(size) {
      // Ensure it's a power of 2
      if ((size & (size - 1)) !== 0) {
        throw new Error("Workgroup size must be a power of 2");
      }
      this.workgroupSize = size;
    }
    
    /**
     * Clean up resources
     */
    destroy() {
      if (this.device) {
        if (typeof this.device.destroy === 'function') {
          this.device.destroy();
        }
        this.device = null;
      }
      this.adapter = null;
      this.computeShader = null;
      this.computePipeline = null;
      this.bindGroupLayout = null;
      this.initialized = false;
    }
  }

/**
 * BamTheta - Binary Angle Measurement implementation
 * Represents angles using 16-bit integers (0-65535) for efficient operations
 */
class BamTheta {
    /**
     * Create a new BamTheta instance
     * @param {number} angleDegrees - Initial angle in degrees
     */
    constructor(angleDegrees = 0) {
      // Convert degrees to BAM units (0-65535)
      this.angle = BamTheta.degreesToBam(angleDegrees);
    }
  
    /**
     * Convert degrees to BAM format
     * @param {number} degrees - Angle in degrees
     * @returns {number} BAM value (0-65535)
     */
    static degreesToBam(degrees) {
      return Math.round((degrees % 360) * 65536 / 360) & 0xFFFF;
    }
  
    /**
     * Convert BAM to degrees
     * @param {number} bam - BAM value (0-65535)
     * @returns {number} Angle in degrees (0-360)
     */
    static bamToDegrees(bam) {
      return (bam * 360) / 65536;
    }
  
    /**
     * Get angle in degrees
     * @returns {number} Angle in degrees (0-360)
     */
    getDegrees() {
      return BamTheta.bamToDegrees(this.angle);
    }
  
    /**
     * Rotate by specified degrees
     * @param {number} angleDegrees - Angle to rotate by in degrees
     * @returns {BamTheta} This instance for chaining
     */
    rotate(angleDegrees) {
      const bamUnits = BamTheta.degreesToBam(angleDegrees);
      this.angle = (this.angle + bamUnits) & 0xFFFF;
      return this;
    }
  
    /**
     * Get sine and cosine of the angle
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {Object} Object with sin and cos properties
     */
    getSinCos(precision = 1000) {
      const radians = (this.angle * 2 * Math.PI) / 65536;
      return {
        sin: Math.round(Math.sin(radians) * precision),
        cos: Math.round(Math.cos(radians) * precision)
      };
    }
  
    /**
     * Get sine of the angle
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {number} Sine value scaled by precision
     */
    getSin(precision = 1000) {
      const radians = (this.angle * 2 * Math.PI) / 65536;
      return Math.round(Math.sin(radians) * precision);
    }
  
    /**
     * Get cosine of the angle
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {number} Cosine value scaled by precision
     */
    getCos(precision = 1000) {
      const radians = (this.angle * 2 * Math.PI) / 65536;
      return Math.round(Math.cos(radians) * precision);
    }
  
    /**
     * Add degrees to the angle
     * @param {number} angleDegrees - Angle to add in degrees
     * @returns {BamTheta} New BamTheta instance
     */
    add(angleDegrees) {
      const result = this.copy();
      result.rotate(angleDegrees);
      return result;
    }
  
    /**
     * Add another BamTheta angle
     * @param {BamTheta} other - Another BamTheta instance
     * @returns {BamTheta} New BamTheta instance
     */
    addBam(other) {
      const result = this.copy();
      result.angle = (result.angle + other.angle) & 0xFFFF;
      return result;
    }
  
    /**
     * Subtract degrees from the angle
     * @param {number} angleDegrees - Angle to subtract in degrees
     * @returns {BamTheta} New BamTheta instance
     */
    subtract(angleDegrees) {
      const result = this.copy();
      result.rotate(-angleDegrees);
      return result;
    }
  
    /**
     * Subtract another BamTheta angle
     * @param {BamTheta} other - Another BamTheta instance
     * @returns {BamTheta} New BamTheta instance
     */
    subtractBam(other) {
      const result = this.copy();
      result.angle = (result.angle - other.angle) & 0xFFFF;
      return result;
    }
  
    /**
     * Compare two BAM angles
     * @param {BamTheta} other - Another BamTheta instance
     * @returns {number} Signed difference (positive if self > other)
     */
    compare(other) {
      let diff = (this.angle - other.angle) & 0xFFFF;
      // Convert to signed (-32768 to 32767)
      if (diff > 32767) {
        diff -= 65536;
      }
      return diff;
    }
  
    /**
     * Test if this angle is greater than another
     * @param {BamTheta} other - Another BamTheta instance
     * @returns {boolean} True if this angle is greater
     */
    isGreaterThan(other) {
      const diff = this.compare(other);
      return 0 < diff && diff < 32768;
    }
  
    /**
     * Get the shortest distance between two angles
     * @param {BamTheta} other - Another BamTheta instance
     * @returns {number} Shortest distance in degrees
     */
    distanceTo(other) {
      const diff = Math.abs(this.compare(other));
      return Math.min(diff, 65536 - diff) * 360 / 65536;
    }
  
    /**
     * Create a copy of this BamTheta
     * @returns {BamTheta} New BamTheta instance
     */
    copy() {
      const result = new BamTheta(0);
      result.angle = this.angle;
      return result;
    }
  
    /**
     * String representation
     * @returns {string} String representation
     */
    toString() {
      return `BamTheta(angle=${this.angle}, degrees=${this.getDegrees().toFixed(2)}°)`;
    }
  }
  
  /**
   * Add two arrays of angles using BamTheta
   * @param {number[]} angles1 - First array of angles in degrees
   * @param {number[]} angles2 - Second array of angles in degrees
   * @returns {number[]} Sum angles in degrees
   */
  function addThetaArrays(angles1, angles2) {
    if (angles1.length !== angles2.length) {
      throw new Error("Arrays must have the same length");
    }
  
    return angles1.map((angle1, i) => {
      const bam1 = new BamTheta(angle1);
      const bam2 = new BamTheta(angles2[i]);
      return bam1.addBam(bam2).getDegrees();
    });
  }
  
  /**
   * Create a lookup table for sine and cosine values
   * @param {number} precision - Precision factor (default=1000)
   * @returns {Object} Lookup tables for sine and cosine
   */
  function createBamLookupTable(precision = 1000) {
    const sinTable = new Int16Array(256);
    const cosTable = new Int16Array(256);
    
    // Create 256-entry table (using 8-bit indices)
    for (let i = 0; i < 256; i++) {
      const angle = (i * 65536) / 256; // Map 0-255 to 0-65535
      const radians = (angle * 2 * Math.PI) / 65536;
      sinTable[i] = Math.round(Math.sin(radians) * precision);
      cosTable[i] = Math.round(Math.cos(radians) * precision);
    }
    
    return { sinTable, cosTable };
  }
  
  /**
   * BamTheta with optimized lookup tables
   */
  class FastBamTheta extends BamTheta {
    /**
     * Initialize lookup tables
     */
    static initialize() {
      if (!FastBamTheta._initialized) {
        const { sinTable, cosTable } = createBamLookupTable(1000);
        FastBamTheta.sinTable = sinTable;
        FastBamTheta.cosTable = cosTable;
        FastBamTheta._initialized = true;
      }
    }
    
    /**
     * Get sine using lookup table
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {number} Sine value
     */
    getSin(precision = 1000) {
      FastBamTheta.initialize();
      // Use high 8 bits as index (256 entries)
      const index = (this.angle >> 8) & 0xFF;
      return (FastBamTheta.sinTable[index] * precision) / 1000;
    }
    
    /**
     * Get cosine using lookup table
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {number} Cosine value
     */
    getCos(precision = 1000) {
      FastBamTheta.initialize();
      // Use high 8 bits as index (256 entries)
      const index = (this.angle >> 8) & 0xFF;
      return (FastBamTheta.cosTable[index] * precision) / 1000;
    }
    
    /**
     * Get sine and cosine using lookup table
     * @param {number} precision - Scaling factor (default=1000)
     * @returns {Object} Object with sin and cos properties
     */
    getSinCos(precision = 1000) {
      return {
        sin: this.getSin(precision),
        cos: this.getCos(precision)
      };
    }
  }
  
  // Static properties for lookup tables
  FastBamTheta._initialized = false;
  FastBamTheta.sinTable = null;
  FastBamTheta.cosTable = null;
  
  // Export the API
  export {
    BamTheta,
    FastBamTheta,
    addThetaArrays
  };
  
  // Export the API
  export {
    BamThetaWebGPU,
    BamTheta
  };