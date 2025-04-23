/**
 * IntTheta2 implementation for WebGPU using 8-bit integers
 * 
 * This class provides methods to represent and manipulate angles
 * using 8-bit integer coordinates (int8) for efficient GPU computation.
 */

// note: seems not working
class IntTheta2WebGPU {
    constructor() {
      this.device = null;
      this.shaderModule = null;
      this.initialized = false;
    }
  
    /**
     * Initialize WebGPU device and create shader module
     * @returns {Promise<IntTheta2WebGPU>} The initialized instance
     */
    async initialize() {
      if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser.");
      }
  
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error("Failed to get GPU adapter.");
      }
  
      this.device = await adapter.requestDevice();
      
      // Create shader module with the IntTheta2 implementation
      this.shaderModule = this.device.createShaderModule({
        code: `
          // Structure to represent IntTheta2 with int8 components
          struct IntTheta2 {
            x: i32, // We'll use only the lower 8 bits (-127 to 127)
            y: i32, // We'll use only the lower 8 bits (-127 to 127)
          }
  
          // Input arrays
          @group(0) @binding(0) var<storage, read> input1: array<IntTheta2>;
          @group(0) @binding(1) var<storage, read> input2: array<IntTheta2>;
          // Output array
          @group(0) @binding(2) var<storage, write> output: array<IntTheta2>;
  
          // CORDIC_ANGLES precomputed (scaled for BAM format)
          const CORDIC_ANGLES = array<i32, 10>(
            11520,  // 45.0 degrees
            6801,   // 26.57 degrees
            3593,   // 14.04 degrees
            1824,   // 7.13 degrees
            916,    // 3.58 degrees
            458,    // 1.79 degrees
            229,    // 0.89 degrees
            115,    // 0.44 degrees
            57,     // 0.22 degrees
            29      // 0.11 degrees
          );
  
          // CORDIC gain factor (scaled by 1024)
          const CORDIC_GAIN: i32 = 607;  // ~= 1/1.647 * 1000
  
          // Fast normalization for int8 coordinates
          fn normalizeInt8(vec: IntTheta2) -> IntTheta2 {
            var result: IntTheta2;
            
            let absX = abs(vec.x);
            let absY = abs(vec.y);
            
            // Quick maximum norm approximation: max + min/2
            let approxLength = max(absX, absY) + (min(absX, absY) >> 1);
            
            if (approxLength > 0) {
              // Scale to keep within int8 range (-127 to 127)
              let scale = 127;
              result.x = (vec.x * scale) / approxLength;
              result.y = (vec.y * scale) / approxLength;
            } else {
              result.x = 0;
              result.y = 0;
            }
            
            // Ensure we're in int8 range
            result.x = clamp(result.x, -127, 127);
            result.y = clamp(result.y, -127, 127);
            
            return result;
          }
  
          // Get BAM representation (Binary Angle Measurement)
          fn getBam(theta: IntTheta2) -> i32 {
            // We're using atan2 approximation
            let absX = abs(theta.x);
            let absY = abs(theta.y);
            
            // Fast atan2 approximation
            var angle: i32;
            if (absX > absY) {
              let t = (absY << 8) / max(absX, 1); // Q8 format
              angle = 256 - (256 * t) / (t * t + 512); // in 1/1024 of circle
            } else {
              let t = (absX << 8) / max(absY, 1); // Q8 format
              angle = 768 - (256 * t) / (t * t + 512); // in 1/1024 of circle
            }
            
            // Adjust for quadrant
            if (theta.x < 0) {
              if (theta.y < 0) {
                angle = 512 - angle; // Third quadrant
              } else {
                angle = 512 + angle; // Second quadrant
              }
            } else if (theta.y < 0) {
              angle = 1024 - angle; // Fourth quadrant
            }
            
            // Convert to 0-65535 BAM range (0-360 degrees)
            return (angle * 64) & 0xFFFF;
          }
  
          // Convert from BAM to IntTheta2
          fn bamToIntTheta2(bam: i32) -> IntTheta2 {
            var result: IntTheta2;
            
            // Use lookup table approach for common angles
            let angleSector = (bam >> 12) & 0xF; // 0-15 for 24-bit sectors
            
            switch(angleSector) {
              case 0: { // 0-22.5 degrees
                result.x = 127;
                result.y = (bam * 127) / 4096;
                break;
              }
              case 1: { // 22.5-45 degrees
                let t = bam - 4096;
                result.x = 127 - (t * 127) / 4096;
                result.y = 127 * t / 4096;
                break;
              }
              case 2: { // 45-67.5 degrees
                let t = bam - 8192;
                result.x = 127 - (t * 127) / 4096;
                result.y = 127;
                break;
              }
              case 3: { // 67.5-90 degrees
                let t = bam - 12288;
                result.x = 0 - (t * 127) / 4096;
                result.y = 127;
                break;
              }
              case 4: { // 90-112.5 degrees
                let t = bam - 16384;
                result.x = -127 * t / 4096;
                result.y = 127;
                break;
              }
              case 5: { // 112.5-135 degrees
                let t = bam - 20480;
                result.x = -127;
                result.y = 127 - (t * 127) / 4096;
                break;
              }
              case 6: { // 135-157.5 degrees
                let t = bam - 24576;
                result.x = -127;
                result.y = 127 - (t * 127) / 4096;
                break;
              }
              case 7: { // 157.5-180 degrees
                let t = bam - 28672;
                result.x = -127;
                result.y = 0 - (t * 127) / 4096;
                break;
              }
              case 8: { // 180-202.5 degrees
                let t = bam - 32768;
                result.x = -127;
                result.y = -127 * t / 4096;
                break;
              }
              case 9: { // 202.5-225 degrees
                let t = bam - 36864;
                result.x = -127 + (t * 127) / 4096;
                result.y = -127;
                break;
              }
              case 10: { // 225-247.5 degrees
                let t = bam - 40960;
                result.x = -127 + (t * 127) / 4096;
                result.y = -127;
                break;
              }
              case 11: { // 247.5-270 degrees
                let t = bam - 45056;
                result.x = 0 + (t * 127) / 4096;
                result.y = -127;
                break;
              }
              case 12: { // 270-292.5 degrees
                let t = bam - 49152;
                result.x = 127 * t / 4096;
                result.y = -127;
                break;
              }
              case 13: { // 292.5-315 degrees
                let t = bam - 53248;
                result.x = 127;
                result.y = -127 + (t * 127) / 4096;
                break;
              }
              case 14: { // 315-337.5 degrees
                let t = bam - 57344;
                result.x = 127;
                result.y = -127 + (t * 127) / 4096;
                break;
              }
              case 15: { // 337.5-360 degrees
                let t = bam - 61440;
                result.x = 127;
                result.y = 0 + (t * 127) / 4096;
                break;
              }
              default: {
                result.x = 127;
                result.y = 0;
              }
            }
            
            return normalizeInt8(result);
          }
  
          // Add two angles using BAM principle
          fn addThetas(a: IntTheta2, b: IntTheta2) -> IntTheta2 {
            // Convert first angle to BAM
            let bamA = getBam(a);
            
            // Convert second angle to BAM
            let bamB = getBam(b);
            
            // Add in BAM space (modulo 65536 for full circle)
            let bamResult = (bamA + bamB) & 0xFFFF;
            
            // Convert back to IntTheta2
            return bamToIntTheta2(bamResult);
          }
  
          // Subtract two angles using BAM principle
          fn subtractThetas(a: IntTheta2, b: IntTheta2) -> IntTheta2 {
            // Convert angles to BAM
            let bamA = getBam(a);
            let bamB = getBam(b);
            
            // Subtract in BAM space (modulo 65536 for full circle)
            let bamResult = (bamA - bamB) & 0xFFFF;
            
            // Convert back to IntTheta2
            return bamToIntTheta2(bamResult);
          }
  
          // Main compute shader function for addition
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            
            // Ensure we don't go out of bounds
            if (idx >= arrayLength(&output)) {
              return;
            }
            
            // Add the two IntTheta2 values
            output[idx] = addThetas(input1[idx], input2[idx]);
          }
        `
      });
      
      this.initialized = true;
      return this;
    }
  
    /**
     * Create a new pipeline for the IntTheta2 operations
     * @returns {GPUComputePipeline} The compute pipeline
     */
    createPipeline() {
      if (!this.initialized) {
        throw new Error("IntTheta2WebGPU not initialized. Call initialize() first.");
      }
      
      const computePipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: this.shaderModule,
          entryPoint: 'main'
        }
      });
      return computePipeline;
    }
  
    /**
     * Add two arrays of angles (in degrees)
     * @param {number[]} thetaArray1 - First array of angles in degrees
     * @param {number[]} thetaArray2 - Second array of angles in degrees
     * @returns {Promise<number[]>} Resulting angles in degrees
     */
    async addThetaArrays(thetaArray1, thetaArray2) {
      if (!this.initialized) {
        throw new Error("IntTheta2WebGPU not initialized. Call initialize() first.");
      }
      
      const length = thetaArray1.length;
      
      if (length !== thetaArray2.length) {
        throw new Error("Input arrays must have the same length");
      }
      
      // Convert degrees arrays to IntTheta2 arrays
      const intTheta1 = new Int32Array(length * 2);
      const intTheta2 = new Int32Array(length * 2);
      
      for (let i = 0; i < length; i++) {
        const theta1 = degreesToIntTheta2(thetaArray1[i]);
        const theta2 = degreesToIntTheta2(thetaArray2[i]);
        
        intTheta1[i*2] = theta1.x;
        intTheta1[i*2 + 1] = theta1.y;
        
        intTheta2[i*2] = theta2.x;
        intTheta2[i*2 + 1] = theta2.y;
      }
      
      // Create buffer for the output
      const outputBuffer = this.device.createBuffer({
        size: length * 2 * 4, // 2 int32 values per theta, 4 bytes per int32
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      });
      
      // Create buffers for input data
      const input1Buffer = this.device.createBuffer({
        size: intTheta1.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Int32Array(input1Buffer.getMappedRange()).set(intTheta1);
      input1Buffer.unmap();
      
      const input2Buffer = this.device.createBuffer({
        size: intTheta2.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
      });
      new Int32Array(input2Buffer.getMappedRange()).set(intTheta2);
      input2Buffer.unmap();
      
      // Create a buffer for reading back the results
      const resultBuffer = this.device.createBuffer({
        size: length * 2 * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      });
      
      // Create the bind group
      const computePipeline = this.createPipeline();
      const bindGroup = this.device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: input1Buffer } },
          { binding: 1, resource: { buffer: input2Buffer } },
          { binding: 2, resource: { buffer: outputBuffer } }
        ]
      });
      
      // Create command encoder and compute pass
      const commandEncoder = this.device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      computePass.setPipeline(computePipeline);
      computePass.setBindGroup(0, bindGroup);
      
      // Dispatch workgroups - one per 64 elements (or fraction thereof)
      const workgroupsNeeded = Math.ceil(length / 64);
      computePass.dispatchWorkgroups(workgroupsNeeded);
      computePass.end();
      
      // Copy the output buffer to the result buffer for reading
      commandEncoder.copyBufferToBuffer(
        outputBuffer, 0,
        resultBuffer, 0,
        length * 2 * 4
      );
      
      // Submit the command buffer
      this.device.queue.submit([commandEncoder.finish()]);
      
      // Read back the results
      await resultBuffer.mapAsync(GPUMapMode.READ);
      const resultData = new Int32Array(resultBuffer.getMappedRange());
      
      // Convert the results back to degrees
      const resultDegrees = new Array(length);
      for (let i = 0; i < length; i++) {
        const x = resultData[i*2];
        const y = resultData[i*2 + 1];
        resultDegrees[i] = intTheta2ToDegrees(x, y);
      }
      
      resultBuffer.unmap();
      
      return resultDegrees;
    }
  }
  
  /**
   * Convert degrees to IntTheta2 format (int8 x,y components)
   * @param {number} degrees - Angle in degrees
   * @returns {Object} Object with x,y properties in int8 range
   */
  function degreesToIntTheta2(degrees) {
    const radians = degrees * (Math.PI / 180);
    // Scale to -127 to 127 range for int8
    const x = Math.round(Math.cos(radians) * 127);
    const y = Math.round(Math.sin(radians) * 127);
    return { x, y };
  }
  
  /**
   * Convert IntTheta2 back to degrees
   * @param {number} x - x component in int8 range
   * @param {number} y - y component in int8 range
   * @returns {number} Angle in degrees (0-360)
   */
  function intTheta2ToDegrees(x, y) {
    const degrees = Math.atan2(y, x) * (180 / Math.PI);
    return (degrees + 360) % 360; // Normalize to 0-360
  }
  
  /**
   * Fast normalization for int8 coordinates
   * @param {number} x - x component
   * @param {number} y - y component
   * @returns {Object} Normalized x,y components
   */
  function normalizeInt8(x, y) {
    const absX = Math.abs(x);
    const absY = Math.abs(y);
    
    // Quick maximum norm approximation: max + min/2
    const approximateLength = Math.max(absX, absY) + (Math.min(absX, absY) >> 1);
    
    if (approximateLength === 0) return { x: 0, y: 0 };
    
    // Scale to keep within int8 range (-127 to 127)
    const scale = 127 / approximateLength;
    return {
      x: Math.round(x * scale),
      y: Math.round(y * scale)
    };
  }
  
  // Export the API
  export {
    IntTheta2WebGPU,
    degreesToIntTheta2,
    intTheta2ToDegrees,
    normalizeInt8
  };