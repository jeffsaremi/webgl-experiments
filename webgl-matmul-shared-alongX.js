async function runMatMulSharedAlongX(gl, texA, texB, width, height, sharedDim, texC) {
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    // Texture samplers
    uniform sampler2D A;
    uniform sampler2D B;

    void main()
    {
      float value = 0.0;
      int x = int(TexCoord.s * ${width}.0); // rescale
      int y = int(TexCoord.t * ${height}.0); // rescale
      // loop over the shared dim
      for(int k=0; k < ${sharedDim}; ++k) {
        float a = texelFetch(A, ivec2(k, y), 0).r;
        float b = texelFetch(B, ivec2(k, x), 0).r;
        value += a * b;
      }
      TexelValue = vec4(value);
    }`;

const program = createProgram(gl, getDefaultVertexShader(gl),
compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');

  gl.useProgram(program);

  attachOutputTexture(gl, texC);
  gl.viewport(0, 0, width, height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};

// CPU Equivalent for result comparison only
function cpuMatMulSharedAlongX(a, shapeA, b, shapeB, c) {
  let offset = 0;
  for (let i = 0; i < shapeA[0]; i++) {
    const aoffset = i * shapeA[1];
    for (let j = 0; j < shapeB[0]; j++) {
      const boffset = j * shapeB[1];
      let sum = 0;
      for (let k = 0; k < shapeA[1]; k++) {
        sum += a[aoffset+ k] * b[boffset+k];
      }
      c[offset] = sum;
      offset++;
    }
  }
}

function getTestData() {
  return [
    { a:[3,8], b:[4,8]},
    { a:[56*56,64], b:[64,64]},
    { a:[56*56,64], b:[256,64]},
    { a:[7*7,256], b:[64,256]},
    { a:[56*56,512], b:[256,512]},
    { a:[28*28,768], b:[128,768]},
    { a:[28*28,2304], b:[128,2304]},
    { a:[14*14,1024], b:[256,1024]},
    { a:[7*7,2048], b:[512,2048]}
  ];
}

//
// Main
//
async function main() {
  const canvas = createCanvas(1, 1);
  const gl = getContext(canvas);
  setupVBO(gl);
  createFrameBuffer(gl);

  const testDatas = getTestData();
  for(let i = 0; i < testDatas.length; ++i) {
    const testData = testDatas[i];
    const sharedDim = testData.a[1];
    const shapeA = testData.a;
    const shapeB = testData.b;
    console.info(`Running matmul-shared-alongX for [${shapeA.toString()}]-[${shapeB.toString()}]`);
    // output texture dimensions
    const width = shapeB[0];
    const height = shapeA[0];

    const a = createRandomArray(shapeA[0] * shapeA[1]);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    //debugPrintTexture(gl, texA, shapeA[1], shapeA[0], gl.RED, gl.FLOAT);
    const b = createRandomArray(shapeB[0] * shapeB[1]);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);
    //debugPrintTexture(gl, texB, shapeB[1], shapeB[0], gl.RED, gl.FLOAT);

    const c = new Float32Array(width * height);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

    console.time('matmul');
    await runMatMulSharedAlongX(gl, texA, texB, width, height, sharedDim, texC);
    console.timeEnd('matmul');
    console.time('readpixels');
    readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
    console.timeEnd('readpixels');
    const expected = new Float32Array(width * height);
    cpuMatMulSharedAlongX(a, shapeA, b, shapeB, expected);
    if(!compareOutputs(c, expected, 0.1)) {
      console.error('Expected and Actual did not match');
      console.log(c);
      console.log(expected);
    } else {
      console.info('Actual and expected matched!');
    }
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();
