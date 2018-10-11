
function createMatMulProgram(gl, width, height, sharedDim) {
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;
            // Texture samplers
            uniform sampler2D A;
            uniform sampler2D B;
            uniform float width;
            uniform float height;
            uniform int startX;
            uniform int startY;

            void main()
            {
              float value = 0.0;
              int x = startX + int(TexCoord.s * width); // rescale
              int y = startY + int(TexCoord.t * height); // rescale
              // loop over the shared dim
              for(int k=0; k < ${sharedDim}; ++k) {
                float a = texelFetch(A, ivec2(k, y), 0).r;
                float b = texelFetch(B, ivec2(x, k), 0).r;
                value += a * b;
              }
              TexelValue = vec4(value);
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;

}
function runMatMul(gl, program, texA, texB, width, height, texC, window) {
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');
  const handleW = gl.getUniformLocation(program, 'width');
  const handleH = gl.getUniformLocation(program, 'height');
  const handleX = gl.getUniformLocation(program, 'startX');
  const handleY = gl.getUniformLocation(program, 'startY');

  gl.useProgram(program);

  attachOutputTexture(gl, texC);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);
  for(let i = 0; i < Math.ceil(width/window[0]); ++i) {
    for(let j = 0; j < Math.ceil(height/window[1]); ++j) {
      const startX = i * window[0];
      const startY = j * window[1];
      const w = Math.min(window[0], width - startX);
      const h = Math.min(window[1], height - startY);
      //console.info(`viewport: ${startX}, ${startY}, ${w}, ${h}`);
      gl.uniform1f(handleW, w);
      gl.uniform1f(handleH, h);
      gl.uniform1i(handleX, startX);
      gl.uniform1i(handleY, startY);
      gl.viewport(startX, startY, w, h);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.flush();
    }
  }
};

// CPU Equivalent for result comparison only
function cpuMatMul(a, shapeA, b, shapeB, c) {
  let offset = 0;
  for (let i = 0; i < shapeA[0]; i++) {
    for (let j = 0; j < shapeB[1]; j++) {
      let sum = 0;
      for (let k = 0; k < shapeA[1]; k++) {
        sum += a[i * shapeA[1] + k] * b[k * shapeB[1] +  j];
      }
      c[offset] = sum;
      offset++;
    }
  }
}

//
// Main
//
function main() {
  const canvas = createCanvas(1, 1);
  const gl = getContext(canvas);
  setupVBO(gl);
  createFrameBuffer(gl);

  for(let dim0 = 200; dim0 < 2000; dim0 += 500) {
    const sharedDim = dim0-100;
    const shapeA = [dim0, sharedDim];
    const shapeB = [sharedDim, dim0+100];
    // const sharedDim = 3;
    // const shapeA = [4,sharedDim];
    // const shapeB = [sharedDim,4];
    console.info(`Running matmul-window for [${shapeA.toString()}]-[${shapeB.toString()}]`);
    const width = shapeB[1];
    const height = shapeA[0];
    const matmulProgram = createMatMulProgram(gl, width, height, sharedDim);


    const a = new Float32Array(new Array(shapeA[0] * shapeA[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const a = new Float32Array([1,2,2,1,1,2,1,2,3,4,2,3]);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    const b = new Float32Array(new Array(shapeB[0] * shapeB[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const b = new Float32Array([1,2,3,3,2,1,3,2,2,3,2,1]);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);

    const c = new Float32Array(width * height);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

    const count = 3;
    const compare = true;
    const windows = [[64,64], [128,128], [256,256]];
    for(let k = 0; k < windows.length; ++k) {
      console.info(`Running with window size: [${windows[k].toString()}]`);
      for (let i = 0; i < count; ++i) {
        console.time('matmul-window');
        runMatMul(gl, matmulProgram, texA, texB, width, height, texC, windows[k]);
        readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
        console.timeEnd('matmul-window');
        if(i===0 && compare) {
          const expected = new Float32Array(width * height);
          cpuMatMul(a, shapeA, b, shapeB, expected);
          if(!compareOutputs(c, expected, 0.1)) {
            console.error('Expected and Actual did not match');
            console.log(c);
            console.log(expected)
          } else {
            console.info('Actual and expected matched!')
          }
        }
      }
    }
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();
