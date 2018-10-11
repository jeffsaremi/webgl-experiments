`use implicit;`

function createMatMulProgram(gl, width, height, originWidth, originHeight, sharedDim) {
  const binCount = (width/originWidth) * (height/originHeight);
  const binSize = Math.ceil(sharedDim/binCount);
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;

            uniform sampler2D A;
            uniform sampler2D B;


            float process(ivec2 coords, int start, int end) {
              float value = 0.0;
              int k = start;
              while(k < end) {
                float a = texelFetch(A, ivec2(k, coords.y), 0).r;
                float b = texelFetch(B, ivec2(coords.x, k), 0).r;
                value += a * b;
                ++k;
              }
              return value;
            }
            void main()
            {
              int x = int(TexCoord.s * ${width}.0); // rescale
              int y = int(TexCoord.t * ${height}.0); // rescale

              int i = x / ${originWidth};
              int j = y / ${originHeight};
              int index = j * (${width}/${originWidth}) + i;
              const int count = ${binSize};
              int start = index * count;
              int end = min(start + count, ${sharedDim});

              ivec2 origCoords = ivec2(x - i * ${originWidth}, y - j * ${originHeight});
              TexelValue = vec4(process(origCoords, start, end));
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;
}
function createCombineProgram(gl, width, height, magWidth, magHeight) {
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;
            uniform sampler2D X;

            void main()
            {
              int x = int(TexCoord.s * ${width}.0); // rescale
              int y = int(TexCoord.t * ${height}.0); // rescale
              vec4 value = vec4(0.0);
              for(int k = 0; k < ${magWidth}/${width}; ++k) {
                for(int l = 0; l < ${magHeight}/${height}; ++l) {
                  ivec2 inputCoords = ivec2(x + k * ${width}, y + l * ${height});
                  value += texelFetch(X, inputCoords, 0);
                }
              }
              TexelValue = value;
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;
}
function runMatMul(gl, texA, texB, width, height, texC, sharedDim, buffer, magFactor, magBuffer) {
  const magWidth = width * magFactor;
  const magHeight = height * magFactor;
  const texMag = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, magWidth, magHeight, null);

  console.time('matmul-parallel'); // exclude the time to create magTexture
  const matmulKey = `matmul-parallel-${magWidth}-${magHeight}-${width}-${height}-${sharedDim}`;
  let program = getProgram(matmulKey);
  if(!program) {
    program = createMatMulProgram(gl, magWidth, magHeight, width, height, sharedDim);
    cacheProgram(matmulKey, program);
  }
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');

  gl.useProgram(program);
  gl.viewport(0, 0, magWidth, magHeight);
  attachOutputTexture(gl, texMag);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);
  //checkError(gl); // make sure we have bound all input/output properly
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.flush();
  //readOutput(gl, magWidth, magHeight, gl.RED, gl.FLOAT, magBuffer);
  //console.timeEnd('matmul-parallel'); // exclude the time to create magTexture
  //console.time('matmul-reduce'); // exclude the time to create magTexture
  //debugPrintTexture(gl, texMag, magWidth, magHeight) ;
  // Combining phase
  const mergeKey = `Merge-${width}-${height}-${magWidth}-${magHeight}`;
  let mergeProgram = getProgram(mergeKey);
  if(!mergeProgram) {
    mergeProgram = createCombineProgram(gl, width, height, magWidth, magHeight);
    cacheProgram(mergeKey, mergeProgram);
  }
  gl.useProgram(mergeProgram);
  gl.viewport(0, 0, width, height);
  attachOutputTexture(gl, texC);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texMag);
  gl.uniform1i(gl.getUniformLocation(mergeProgram, 'X'), 0);
  //checkError(gl); // make sure we have bound all input/output properly
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.flush();
  readOutput(gl, width, height, gl.RED, gl.FLOAT, buffer);
  console.timeEnd('matmul-parallel');
  gl.deleteTexture(texMag);
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
    console.info(`Running matmul-parallel for [${shapeA.toString()}]-[${shapeB.toString()}]`)
    const width = shapeB[1];
    const height = shapeA[0];

    const a = new Float32Array(new Array(shapeA[0] * shapeA[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const a = new Float32Array([1,2,2,1,1,2, 0, 1]);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    const b = new Float32Array(new Array(shapeB[0] * shapeB[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const b = new Float32Array([1,2,3,3,2,1, 3, 1]);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);

    const c = new Float32Array(width * height);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

    const count = 3;
    const compare = false;
    const magFactor = 2;
    //for(let magFactor = 2; magFactor < 6; ++magFactor) {
      console.log(`Starting MagFactor = ${magFactor}`);
      const magBuffer = new Float32Array(width * height * magFactor * magFactor);
      for (let i = 0; i < count; ++i) {
        runMatMul(gl, texA, texB, width, height, texC, sharedDim, c, magFactor, magBuffer);

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
      console.log(`Ending MagFactor = ${magFactor}`);
    //}
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();