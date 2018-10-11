
function createMatMulProgram(gl, width, height, sharedDim) {
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;
            // Texture samplers
            uniform sampler2D A;
            uniform sampler2D B;
            uniform sampler2D C;
            uniform mediump int k;

            void main()
            {
              int x = int(TexCoord.s * ${width}.0); // rescale
              int y = int(TexCoord.t * ${height}.0); // rescale
              float value = texture(C, TexCoord).r;
              float a = texelFetch(A, ivec2(k, y), 0).r;
              float b = texelFetch(B, ivec2(x, k), 0).r;
              value += a * b;
              TexelValue = vec4(value);
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;

}
function runMatMul(gl, program, texA, texB, width, height, texC, sharedDim) {
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');
  const handleC = gl.getUniformLocation(program, 'C');
  const handleK = gl.getUniformLocation(program, 'k');

  const texC2 = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);
  gl.useProgram(program);
  gl.viewport(0, 0, width, height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);
  
  const texPair = [texC, texC2];
  let texOut, texIn;
  for(let k = 0; k < sharedDim; ++k) {
    const outIndex = k % 2;
    texOut = texPair[outIndex];
    texIn = texPair[1 - outIndex];
    attachOutputTexture(gl, texOut);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, texIn);
    gl.uniform1i(handleC, 2);
    // bind K
    gl.uniform1i(handleK, k);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();  
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
  const programCache = new Map();
  const shapeA = [1000, 500];
  const shapeB = [500, 2000];
  console.info(`Running matmul=reduce for [${shapeA.toString()}]-[${shapeB.toString()}]`)
  const width = shapeB[1];
  const height = shapeA[0];
  const sharedDim = shapeA[1];
  const matmulProgram = createMatMulProgram(gl, width, height, sharedDim);
  setupVBO(gl);
  createFrameBuffer(gl);

  const a = new Float32Array(new Array(shapeA[0] * shapeA[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
  //const a = new Float32Array([1,2,2,1,1,2]);
  const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
  const b = new Float32Array(new Array(shapeB[0] * shapeB[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
  //const b = new Float32Array([1,2,3,3,2,1]);
  const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);

  const c = new Float32Array(width * height);
  const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

  const count = 5;
  const compare = true;
  let texOut;
  for (let i = 0; i < count; ++i) {
    console.time('matmul-reduce');
    runMatMul(gl, matmulProgram, texA, texB, width, height, texC, sharedDim);
    readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
    console.timeEnd('matmul-reduce');
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

main();