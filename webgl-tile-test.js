
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
async function runDetileProgram(gl, tiledTexture, originalTexture, originalWidth, originalHeight, tileLength, tiledWidth, tiledHeight) {
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
    ${glslOriginalCoordsToTiled()}
    void main() {
      vec2 coords = originalCoordsToTiled(TexCoords, width, height, tileLength, tiledWidth, tiledHeight);
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

  attachOutputTexture(gl, originalTexture);
  gl.viewport(0, 0, originalWidth, originalHeight);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tiledTexture);
  gl.uniform1i(hA, 0);
  gl.uniform1i(hTileLength, tileLength);
  gl.uniform1i(hTiledHeight, tiledHeight);
  gl.uniform1i(hTiledWidth, tiledWidth);
  gl.uniform1i(hHeight, originalHeight);
  gl.uniform1i(hWidth, originalWidth);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};

function calcTiledTextureShape(shape, pivot, tileLength) {
  const leftDimsLength = shape.slice(0, pivot);
  const rightDimsLength = shape.slice(pivot).reduce((a,b)=>a*b);
  const width = Math.ceil(rightDimsLength/tileLength);
  const height = leftDimsLength * tileLength;
  return [ width, height ];
}
function getTestData() {
  return [
    { shape:[3,12]}
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
    const inputShape = testData.shape;
    const tileLength = 2;
    const [tiledWidth, tiledHeight] = calcTiledTextureShape(inputShape, 1, tileLength);
    console.log(`tiledWidth=${tiledWidth}, tiledHeight=${tiledHeight}`);

    const input = new Float32Array(new Array(inputShape[0] * inputShape[1]).fill(0).map((v,i)=>i));
    
    const inputTexture = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, inputShape[1], inputShape[0], input);
    debugPrintTexture(gl, inputTexture, inputShape[1], inputShape[0], gl.RED, gl.FLOAT);

    const tiledTexture = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, tiledWidth, tiledHeight, null);
    await runTileUpProgram(gl, inputTexture, tiledTexture, inputShape[1], inputShape[0], tileLength, tiledWidth, tiledHeight);
    debugPrintTexture(gl, tiledTexture, tiledWidth, tiledHeight, gl.RED, gl.FLOAT);

    const originalWidth = inputShape[1];
    const originalHeight = inputShape[0];
    const originalTexture = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, originalWidth, originalHeight, null);
    await runDetileProgram(gl, tiledTexture, originalTexture, originalWidth, originalHeight, tileLength, tiledWidth, tiledHeight);
    const output = new Float32Array(originalWidth * originalHeight);
    readOutput(gl, originalWidth, originalHeight, gl.RED, gl.FLOAT, output);
    if(!compareOutputs(output, input, 0.1)) {
      console.error('Expected and Actual did not match');
      console.log(output);
      console.log(input);
    } else {
      console.info('Actual and expected matched!');
    }
    gl.deleteTexture(inputTexture);
    gl.deleteTexture(tiledTexture);
    gl.deleteTexture(originalTexture);
  }
}

main();
