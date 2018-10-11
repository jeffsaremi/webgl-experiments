
function createMatMulProgram(gl, tileWidth, tileHeight, sharedDim, tileCount) {
  const valueLines = [];
  for(let i = 0; i < tileCount; ++i) {
    valueLines.push(`TexelValue[${i}] += valueA * texelFetch(B, ivec2(x + ${tileWidth*i}, k), 0).r;`);
  }
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;

            uniform sampler2D A;
            uniform sampler2D B;

            layout(location = 0) out vec4 TexelValue[${tileCount}];

            void main()
            {
              int x = int(TexCoord.s * ${tileWidth}.0); // rescale
              int y = int(TexCoord.t * ${tileHeight}.0); // rescale
              for(int k=0; k < ${sharedDim}; ++k) {
                float valueA = texelFetch(A, ivec2(k, y), 0).r;
                ${valueLines.join('\n')}
              }
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;
}
function createCombineProgram(gl, width, height, tileCount) {
  const tileWidth = Math.ceil(width / tileCount);
  const valueLines = [];
  for(let i=0; i < tileCount; ++i) {
    valueLines.push(`
    if(i == ${i}) {
      TexelValue = texelFetch(Tiles[${i}], tileCoords, 0);
      return;
    }
    `);
  }
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;
            uniform sampler2D Tiles[${tileCount}];

            void main()
            {
              int x = int(TexCoord.s * ${width}.0); // rescale
              int y = int(TexCoord.t * ${height}.0); // rescale
              int i = x / ${tileWidth};
              ivec2 tileCoords = ivec2(x - i * ${tileWidth}, y);
              ${valueLines.join('\n')}
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;
}
async function runMatMul(gl, texA, texB, width, height, sharedDim, textures) {
  const tileWidth = Math.ceil(width / textures.length);
  const tileHeight = height;

  const matmulKey = `matmul-multi-output-${width}-${height}-${textures.length}-${sharedDim}`;
  let program = getProgram(matmulKey);
  if(!program) {
    program = createMatMulProgram(gl, tileWidth, height, sharedDim, textures.length);
    cacheProgram(matmulKey, program);
  }
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');

  gl.useProgram(program);
  gl.viewport(0, 0, tileWidth, tileHeight);
  const drawBuffers = [];
  for (let i = 0; i < textures.length; ++i) {
    gl.bindTexture(gl.TEXTURE_2D, textures[i]);
    // attach texture to framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i,
                            gl.TEXTURE_2D, textures[i], 0);
    drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
  }

  gl.drawBuffers(drawBuffers);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);
  //checkError(gl); // make sure we have bound all input/output properly
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
}
async function runCombine(gl, textures, width, height, texC) {  
  // Combining phase
  const mergeKey = `Merge-${width}-${height}-${textures.length}`;
  let mergeProgram = getProgram(mergeKey);
  if(!mergeProgram) {
    mergeProgram = createCombineProgram(gl, width, height, textures.length);
    cacheProgram(mergeKey, mergeProgram);
  }
  gl.useProgram(mergeProgram);
  gl.viewport(0, 0, width, height);
  //gl.clearBufferiv(gl.COLOR, 0, drawBuffers);
  createFrameBuffer(gl);
  attachOutputTexture(gl, texC);
  const uniformIndices = [];
  for (let i = 0; i < textures.length; ++i) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, textures[i]);
    uniformIndices.push(i);
  }
  gl.uniform1iv(gl.getUniformLocation(mergeProgram, 'Tiles[0]'), uniformIndices);
  //checkError(gl); // make sure we have bound all input/output properly
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
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
async function main() {
  const canvas = createCanvas(1, 1);
  const gl = getContext(canvas);
  setupVBO(gl);
  createFrameBuffer(gl);

  for(let dim0 = 200; dim0 < 2000; dim0 += 500) {
    const sharedDim = dim0-100;
    const shapeA = [dim0, sharedDim];
    const shapeB = [sharedDim, dim0+100];

    console.info(`Running matmul-multi-output for [${shapeA.toString()}]-[${shapeB.toString()}]`)
    const width = shapeB[1];
    const height = shapeA[0];


    const a = new Float32Array(new Array(shapeA[0] * shapeA[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const a = new Float32Array([1,2,3,4,5,6,6,5,4,3,2,1,1,3,5,2,4,6,2,4,6,1,3,5]);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    const b = new Float32Array(new Array(shapeB[0] * shapeB[1]).fill(0).map(v=>Math.floor(Math.random()*10)));
    //const b = new Float32Array([6,4,2,1,3,5,1,3,5,2,4,6,1,2,3,3,2,1,6,5,4,4,5,6]);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);

    const c = new Float32Array(width * height);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);


    const count = 5;
    const compare = true;
    const dops = [2,4,8];
    for(let j = 0; j < dops.length; ++j) {
      const tileWidth = Math.ceil(width / dops[j]);
      const textures = [];
      for (let i = 0; i < dops[j]; ++i) {
        const tex = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, tileWidth, height, null);
        textures.push(tex);
      }
      console.log(`Running with degree of parallelism: ${dops[j]}`);
      for (let i = 0; i < count; ++i) {
        console.time('matmul-multi');
        await runMatMul(gl, texA, texB, width, height, sharedDim, textures);
        console.timeEnd('matmul-multi');
        console.time('combine');
        await runCombine(gl, textures, width, height, texC);
        console.timeEnd('combine');
        console.time('readpixels');
        readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
        console.timeEnd('readpixels');
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
      for (let i = 0; i < dops[j]; ++i) {
        gl.deleteTexture(textures[i]);
      }
    }
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();