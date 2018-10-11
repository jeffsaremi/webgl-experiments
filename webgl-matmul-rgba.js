function createFromRGBAProgram(gl, width, height) {
    const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    // Texture samplers
    uniform sampler2D A;

    void main()
    {
        int x = int(TexCoord.s * ${width}.0); // rescale
        int y = int(TexCoord.t * ${height}.0); // rescale
        
        int srcX = x / 4;
        int channel = x  - srcX * 4;
        vec4 value = texelFetch(A, ivec2(srcX, y), 0);
        if(channel == 0) {
            TexelValue = vec4(value.r);
        } else if(channel == 1) {
            TexelValue = vec4(value.g);
        } else if(channel == 2) {
            TexelValue = vec4(value.b);
        } else {
            TexelValue = vec4(value.a);
        }
    }`;
    const program = createProgram(
        gl, getDefaultVertexShader(gl),
        compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    return program;
}
function fromRGBA(gl, texSrc, texDest, width, height) {
    const fromRGBAKey =
        `fromRGBA-${width}-${height}`;
    let program = getProgram(fromRGBAKey);
    if (!program) {
        program = createFromRGBAProgram(gl, width, height);
        cacheProgram(fromRGBAKey, program);
    }
    const handleA = gl.getUniformLocation(program, 'A');

    gl.useProgram(program);

    // Create and bind a framebuffer
    attachOutputTexture(gl, texDest);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texSrc);
    gl.uniform1i(handleA, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
}
function createToRGBAProgram(gl, width, height, srcWidth, srcHeight, axis) {
    const remainderLine = axis ===0 ? `int remainder = ${srcWidth} -  x;` : `int remainder = ${srcHeight} -  y;`;
    const ivecPlus1 = axis ===0 ? `ivec2 up1 = ivec2(1,0);` : `ivec2 up1 = ivec2(0,1);`;
    const ivecPlus2 = axis ===0 ? `ivec2 up2 = ivec2(2,0);` : `ivec2 up2 = ivec2(0,2);`;
    const ivecPlus3 = axis ===0 ? `ivec2 up3 = ivec2(3,0);` : `ivec2 up3 = ivec2(0,3);`;
    const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    // Texture samplers
    uniform sampler2D A;

    void main()
    {
        int x = int(TexCoord.s * ${width}.0); // rescale
        int y = int(TexCoord.t * ${height}.0); // rescale
        ${remainderLine}
        ${ivecPlus1}
        ${ivecPlus2}
        ${ivecPlus3}
        ivec2 xy = ivec2(x,y);
        if(remainder >= 4) {
            TexelValue = vec4(
                texelFetch(A, xy, 0).r,
                texelFetch(A, xy + up1, 0).r,
                texelFetch(A, xy + up2, 0).r,
                texelFetch(A, xy + up3, 0).r);
        } else if(remainder == 3) {
            TexelValue = vec4(
                texelFetch(A, xy, 0).r,
                texelFetch(A, xy + up1, 0).r,
                texelFetch(A, xy + up2, 0).r,
                0.0);
        } else if(remainder == 2) {
            TexelValue = vec4(
                texelFetch(A, xy, 0).r,
                texelFetch(A, xy + up1, 0).r,
                0.0,
                0.0);
        } else {
            TexelValue = vec4(
                texelFetch(A, xy, 0).r,
                0.0,
                0.0,
                0.0);
        }
    }`;
    const program = createProgram(
        gl, getDefaultVertexShader(gl),
        compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    return program;
}
function toRGBA(gl, texSrc, texDest, width, height, srcWidth, srcHeight, axis) {
    const toRGBAKey =
        `toRGBA-${width}-${height}-${srcWidth}-${srcHeight}-${axis}`;
    let program = getProgram(toRGBAKey);
    if (!program) {
        program = createToRGBAProgram(gl, width, height, srcWidth, srcHeight, axis);
        cacheProgram(toRGBAKey, program);
    }
    const handleA = gl.getUniformLocation(program, 'A');

    gl.useProgram(program);

    // Create and bind a framebuffer
    attachOutputTexture(gl, texDest);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texSrc);
    gl.uniform1i(handleA, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.flush();
}
function createMatMulProgram(gl, width, height, sharedDim) {
    const adjustedSharedDim = Math.ceil(sharedDim / 4);
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
                              for(int k=0; k < ${adjustedSharedDim}; ++k) {
                                  vec4 a = texelFetch(A, ivec2(k, x), 0);
                                  vec4 b = texelFetch(B, ivec2(y, k), 0);
                                  value += dot(a, b);
                              }
                              TexelValue = vec4(value);
                          }`;
  
    const program = createProgram(gl, getDefaultVertexShader(gl),
        compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    return program;
  }
function runMatMul(gl, texA, texB, width, height, texC, sharedDim) {

    const matmulKey =
        `matmul-matmul-RGBA-${width}-${height}-${sharedDim}`;
    let program = getProgram(matmulKey);
    if (!program) {
        program = createMatMulProgram(gl, width, height, sharedDim);
        cacheProgram(matmulKey, program);
    }
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
    gl.flush();
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

    //for (let dim0 = 200; dim0 < 2000; dim0 += 500) {
        // const sharedDim = dim0 - 100;
        // const shapeA = [dim0, sharedDim];
        // const shapeB = [sharedDim, dim0 + 100];
        const sharedDim = 3;
        const shapeA = [2, sharedDim];
        const shapeB = [sharedDim, 2];
        console.info(
            `Running matmul for [${shapeA.toString()}]-[${shapeB.toString()}]`);
        const width = shapeB[1];
        const height = shapeA[0];

        // const a = new Float32Array(new Array(shapeA[0] * shapeA[1])
        //                                 .fill(0)
        //                                 .map(v => Math.floor(Math.random() * 10)));
        const a = new Float32Array([1,2,2,1,1,2]);
        const texA =
            createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
        const texAPrime =
            createTexture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, Math.ceil(shapeA[1]/4), shapeA[0], null);
        // const b = new Float32Array(new Array(shapeB[0] * shapeB[1])
        //                                 .fill(0)
        //                                 .map(v => Math.floor(Math.random() * 10)));
        const b = new Float32Array([1,2,3,3,2,1]);
        const texB =
            createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);
        const texBPrime =
            createTexture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, shapeB[1], Math.ceil(shapeB[0]/4), null);
        const c = new Float32Array(width * height);
        // const texCPrime =
        //     createTexture(gl, gl.RGBA32F, gl.RGBA, gl.FLOAT, Math.ceil(width/4), height, null);
        const texC =
            createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

        const count = 1;
        const compare = true;
        for (let i = 0; i < count; ++i) {
            console.time('matmul-RGBA');
            toRGBA(gl, texA, texAPrime, Math.ceil(shapeA[1]/4), shapeA[0], shapeA[1], shapeA[0], 0);
            debugPrintTexture(gl, texAPrime, Math.ceil(shapeA[1]/4), shapeA[0], gl.RGBA, gl.FLOAT);
            toRGBA(gl, texB, texBPrime, shapeB[1], Math.ceil(shapeB[0]/4), shapeB[1], shapeB[0], 1);
            debugPrintTexture(gl, texBPrime, shapeB[1], Math.ceil(shapeB[0]/4), gl.RGBA, gl.FLOAT);
            runMatMul(gl, texAPrime, texBPrime, width, height, texC, sharedDim);
            //debugPrintTexture(gl, texC, width, height, gl.RED, gl.FLOAT);
            //fromRGBA(gl, texCPrime, texC, width, height);
            readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
            console.timeEnd('matmul-RGBA');
            if (i === 0 && compare) {
            const expected = new Float32Array(width * height);
            cpuMatMul(a, shapeA, b, shapeB, expected);
            if (!compareOutputs(c, expected, 0.1)) {
                console.error('Expected and Actual did not match');
                console.log(c);
                console.log(expected)
            } else {
                console.info('Actual and expected matched!')
            }
            }
        }
        gl.deleteTexture(texA);
        gl.deleteTexture(texB);
        gl.deleteTexture(texC);
    //}
}

main();
