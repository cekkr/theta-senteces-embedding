# Implementazione di IntTheta2 per WebGPU

Questa implementazione trasforma l'algoritmo IntTheta2 originale in una versione WebGPU che utilizza interi a 8 bit (`int8`) per rappresentare le coordinate x e y di ogni angolo.

## Panoramica dell'Implementazione

### Rappresentazione degli Angoli

Nel codice originale, IntTheta2 memorizza gli angoli utilizzando coordinate intere di precisione arbitraria (controllata dal parametro `precision`). Nell'implementazione WebGPU, gli angoli sono scalati per adattarsi al formato `int8` che ha un intervallo da -128 a 127 (anche se usiamo solo -127 a 127 per semplificare la normalizzazione).

### Funzionalità Implementate

1. **Conversione tra gradi e IntTheta2**: Funzioni per convertire angoli in gradi in coordinate IntTheta2 e viceversa.
2. **Normalizzazione**: Adattamento della funzione `fast_normalize()` per lavorare con interi a 8 bit.
3. **Rotazione**: Implementazione della rotazione tramite manipolazione nel formato BAM (Binary Angle Measurement).
4. **Somma di angoli**: Un kernel WebGPU per sommare due array di angoli.

## Dettagli Tecnici

### 1. Rappresentazione in WebGPU

Nel shader WGSL, definiamo una struttura per rappresentare gli angoli IntTheta2:

```wgsl
struct IntTheta2 {
  x: i32, // Usiamo solo i primi 8 bit inferiori
  y: i32, // Usiamo solo i primi 8 bit inferiori
}
```

Anche se il tipo è `i32`, usiamo solo l'intervallo da -127 a 127 per mantenere la compatibilità con `int8`.

### 2. Conversione e Normalizzazione

La conversione tra gradi e IntTheta2 è implementata sia in JavaScript che nel shader:

```javascript
// Da gradi a IntTheta2
function degreesToIntTheta2(degrees) {
  const radians = degrees * (Math.PI / 180);
  // Scala nell'intervallo -127 a 127 per int8
  const x = Math.round(Math.cos(radians) * 127);
  const y = Math.round(Math.sin(radians) * 127);
  return { x, y };
}

// Da IntTheta2 a gradi
function intTheta2ToDegrees(x, y) {
  const degrees = Math.atan2(y, x) * (180 / Math.PI);
  return (degrees + 360) % 360; // Normalizza a 0-360
}
```

La normalizzazione veloce degli interi a 8 bit utilizza l'approssimazione della norma massima:

```wgsl
fn normalizeInt8(vec: IntTheta2) -> IntTheta2 {
  var result: IntTheta2;
  
  let absX = abs(vec.x);
  let absY = abs(vec.y);
  
  // Approssimazione rapida della norma: max + min/2
  let approxLength = max(absX, absY) + (min(absX, absY) >> 1);
  
  if (approxLength > 0) {
    // Scala per mantenere nell'intervallo int8 (-127 a 127)
    let scale = 127;
    result.x = (vec.x * scale) / approxLength;
    result.y = (vec.y * scale) / approxLength;
  } else {
    result.x = 0;
    result.y = 0;
  }
  
  // Assicura che siamo nell'intervallo int8
  result.x = clamp(result.x, -127, 127);
  result.y = clamp(result.y, -127, 127);
  
  return result;
}
```

### 3. Rappresentazione BAM

Il formato BAM (Binary Angle Measurement) rappresenta gli angoli come interi tra 0 e 65535, dove 0 = 0° e 65535 = 359.99°. Questo facilita le operazioni di rotazione e somma: