async function runDotProdTiled(gl, texA, texB, texAWidth, textAHeight, texBWidth, textBHeight, width, height, sharedDim, tileLength, texC) {
  const fragmentShaderSource = `
  precision highp float;
  precision mediump int;
  varying vec2 TexCoords;
  // Texture samplers
  uniform sampler2D A;
  uniform sampler2D B;
  // uniform int AWidth;
  // uniform int AHeight;
  // uniform int BWidth;
  // uniform int BHeight;

  const int tileLength = ${tileLength};
  const int sharedDim = ${sharedDim};
  ivec2 leftWH = ivec2(${texAWidth},${textAHeight});
  ivec2 rightWH = ivec2(${texBWidth},${textBHeight});

  vec4 texel(sampler2D texture, ivec2 xy, int width, int height) {
    vec2 coords = (vec2(xy) + vec2(0.5,0.5)) / vec2(width, height);
    return texture2D(texture, coords);
  }

  float _2by2DotProd(sampler2D left, sampler2D right, ivec2 leftCoords, ivec2 rightCoords) {
    float sum = 0.0;
    sum += texel(left, leftCoords, leftWH[0], leftWH[1]).r            * texel(right, rightCoords, rightWH[0], rightWH[1]).r;
    sum += texel(left, leftCoords+ivec2(1,0), leftWH[0], leftWH[1]).r * texel(right, rightCoords+ivec2(1,0), rightWH[0], rightWH[1]).r;
    sum += texel(left, leftCoords+ivec2(0,1), leftWH[0], leftWH[1]).r * texel(right, rightCoords+ivec2(0,1), rightWH[0], rightWH[1]).r;
    sum += texel(left, leftCoords+ivec2(1,1), leftWH[0], leftWH[1]).r * texel(right, rightCoords+ivec2(1,1), rightWH[0], rightWH[1]).r;
    return sum;
  }
  float tileDotProd(sampler2D left, sampler2D right, int leftBandIndex, int rightBandIndex, int tileIndex) {
    ivec2 lcoords = ivec2(tileIndex, leftBandIndex) * ivec2(tileLength, tileLength);
    ivec2 rcoords = ivec2(tileIndex, rightBandIndex) * ivec2(tileLength, tileLength);
    // float sum = 0.0;
    // for(int i=0; i < tileLength/2; i += 2) {
    //   lcoords += ivec2(i*2, 0);
    //   rcoords += ivec2(0, i*2);
    //   for(int j=0; j < tileLength/2; j += 2) {
    //     lcoords += ivec2(0, j*2);
    //     rcoords += ivec2(j*2, 0);
    //     sum += _2by2DotProd(left, right, lcoords, rcoords);
    //   }
    // }
    // return sum;
    return _2by2DotProd(left, right, lcoords, rcoords);
  }
  float bandDotProd(sampler2D left, sampler2D right, int leftBandIndex, int rightBandIndex) {
    const int tileCount = sharedDim / (tileLength*tileLength);
    float sum = 0.0;
    for(int k = 0; k < tileCount; ++k) {
      sum += tileDotProd(left, right, leftBandIndex, rightBandIndex, k);
    }
    return sum;
  }
  void main()
  {
    float value = 0.0;
    int x = int(TexCoords.s * ${width}.0); // rescale
    int y = int(TexCoords.t * ${height}.0); // rescale
    // loop over the shared dim
    value = bandDotProd(A, B, y, x);
    gl_FragColor = vec4(value);
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

async function runTileUpProgram(gl, inputTexture, tiledTexture, originalWidth, originalHeight, tileLength, tiledWidth, tiledHeight) {
  const fragmentShaderSource = `
    precision highp float;
    precision mediump int;
    varying vec2 TexCoords;
    uniform sampler2D A;

    uniform int tileLength;
    uniform int tiledHeight;
    uniform int tiledWidth;
    uniform int height;
    uniform int width;

    ${glslOffsetToCoords()}
    ${glslCoordsToOffset()}
    ${glslTiledCoordsToOriginal()}
    void main() {
      vec2 coords = tiledCoordsToOriginal(TexCoords, tiledWidth, tiledHeight, tileLength, width, height);
      gl_FragColor = texture2D(A, coords);
    }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  const hA = gl.getUniformLocation(program, 'A');
  const hTileLength = gl.getUniformLocation(program, 'tileLength');
  const hTiledHeight = gl.getUniformLocation(program, 'tiledHeight');
  const hTiledWidth = gl.getUniformLocation(program, 'tiledWidth');
  const hHeight = gl.getUniformLocation(program, 'height');
  const hWidth = gl.getUniformLocation(program, 'width');

  gl.useProgram(program);

  attachOutputTexture(gl, tiledTexture);
  gl.viewport(0, 0, tiledWidth, tiledHeight);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, inputTexture);
  gl.uniform1i(hA, 0);
  gl.uniform1i(hTileLength, tileLength);
  gl.uniform1i(hTiledHeight, tiledHeight);
  gl.uniform1i(hTiledWidth, tiledWidth);
  gl.uniform1i(hHeight, originalHeight);
  gl.uniform1i(hWidth, originalWidth);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};
async function runStraightCopyProgram(gl, inputTexture, outputTexture, width, height) {
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    precision mediump int;
    in vec2 TexCoord;
    out vec4 TexelValue;
    uniform sampler2D A;

    void main() {
      TexelValue = texture(A, TexCoord);
    }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  const hA = gl.getUniformLocation(program, 'A');

  gl.useProgram(program);

  attachOutputTexture(gl, outputTexture);
  gl.viewport(0, 0, width, height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, inputTexture);
  gl.uniform1i(hA, 0);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};

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
    console.info(`Running dotProduct for [${shapeA.toString()}]-[${shapeB.toString()}]`);

    const tileLength = 2;

    const a = createRandomArray(shapeA[0] * shapeA[1]);
    const texA = createStorage(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    // const texCopyA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], null);
    // await runStraightCopyProgram(gl, texA, texCopyA, shapeA[1], shapeA[0]);
    //debugPrintTexture(gl, texA, shapeA[1], shapeA[0], gl.RED, gl.FLOAT);

    const [tiledWidthA, tiledHeightA] = [Math.ceil(shapeA[1]/tileLength), shapeA[0]*tileLength];
    const tiledTexA = createStorage(gl, gl.R32F, gl.RED, gl.FLOAT, tiledWidthA, tiledHeightA, null);
    console.time('tiledUpA');
    await runTileUpProgram(gl, texA, tiledTexA, shapeA[1], shapeA[0], tileLength, tiledWidthA, tiledHeightA);
    console.timeEnd('tiledUpA');
    //debugPrintTexture(gl, tiledTexA, tiledWidthA, tiledHeightA, gl.RED, gl.FLOAT);

    const b = createRandomArray(shapeB[0] * shapeB[1]);
    const texB = createStorage(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);
    const [tiledWidthB, tiledHeightB] = [Math.ceil(shapeB[1]/tileLength), shapeB[0]*tileLength];
    const tiledTexB = createStorage(gl, gl.R32F, gl.RED, gl.FLOAT, tiledWidthB, tiledHeightB, null);
    console.time('tiledUpB');
    await runTileUpProgram(gl, texB, tiledTexB, shapeB[1], shapeB[0], tileLength, tiledWidthB, tiledHeightB);
    console.timeEnd('tiledUpB');
    //debugPrintTexture(gl, tiledTexB, tiledWidthB, tiledHeightB, gl.RED, gl.FLOAT);
    // output texture dimensions
    const width = shapeB[0];
    const height = shapeA[0];
    const c = new Float32Array(width * height);
    const texC = createStorage(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

    console.time('tiledDotProd');
    await runDotProdTiled(gl, tiledTexA, tiledTexB, tiledWidthA, tiledHeightA, tiledWidthB, tiledHeightB, width, height, sharedDim, tileLength, texC);
    console.timeEnd('tiledDotProd');
    console.time('readpixels');
    readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
    console.timeEnd('readpixels');
    const expected = new Float32Array(width * height);
    console.time('cpumatmul');
    cpuDotProd(a, shapeA, b, shapeB, expected);
    console.timeEnd('cpumatmul');
    if(!compareOutputs(c, expected, 0.1)) {
      console.error('Expected and Actual did not match');
      console.log(c);
      console.log(expected);
      const leftBuffer  = simConvertOriginalToTiled(a, shapeA[1], shapeA[0], tileLength);
      //console.log('leftBuffer', leftBuffer);
      const rightBuffer = simConvertOriginalToTiled(b, shapeB[1], shapeB[0], tileLength);
      //console.log('rightBuffer', rightBuffer);
      const simResults = simTiledDotProduct(leftBuffer, tiledWidthA, tiledHeightA, rightBuffer, tiledWidthB, tiledHeightB, sharedDim, tileLength, width, height);
      console.log('SimResults', simResults);
  
    } else {
      console.info('Actual and expected matched!');
    }

    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(tiledTexA);
    gl.deleteTexture(tiledTexB);
    gl.deleteTexture(texC);
  }
}

main();
