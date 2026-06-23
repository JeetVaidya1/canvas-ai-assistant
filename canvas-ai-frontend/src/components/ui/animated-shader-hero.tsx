import { useRef, useEffect } from 'react'

const defaultShaderSource = `#version 300 es
precision highp float;
out vec4 O;
uniform vec2 resolution;
uniform float time;
#define FC gl_FragCoord.xy
#define T time
#define R resolution
#define MN min(R.x,R.y)
float rnd(vec2 p) {
  p=fract(p*vec2(12.9898,78.233));
  p+=dot(p,p+34.56);
  return fract(p.x*p.y);
}
float noise(in vec2 p) {
  vec2 i=floor(p), f=fract(p), u=f*f*(3.-2.*f);
  float
  a=rnd(i),
  b=rnd(i+vec2(1,0)),
  c=rnd(i+vec2(0,1)),
  d=rnd(i+1.);
  return mix(mix(a,b,u.x),mix(c,d,u.x),u.y);
}
float fbm(vec2 p) {
  float t=.0, a=1.; mat2 m=mat2(1.,-.5,.2,1.2);
  for (int i=0; i<5; i++) {
    t+=a*noise(p);
    p*=2.*m;
    a*=.5;
  }
  return t;
}
float clouds(vec2 p) {
    float d=1., t=.0;
    for (float i=.0; i<3.; i++) {
        float a=d*fbm(i*10.+p.x*.2+.2*(1.+i)*p.y+d+i*i+p);
        t=mix(t,d,a);
        d=a;
        p*=2./(i+1.);
    }
    return t;
}
void main(void) {
    vec2 uv=(FC-.5*R)/MN,st=uv*vec2(2,1);
    float bg=clouds(vec2(st.x+T*.15,-st.y));
    vec3 col=vec3(bg*.04,bg*.10,bg*.22);
    float d=length(uv);
    col*=1.0-d*0.4;
    col+=.015/(d+.4)*vec3(.05,.14,.22);
    O=vec4(col,1);
}`

export function useShaderBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationFrameRef = useRef<number>(0)

  useEffect(() => {
    if (!canvasRef.current) return

    const canvas = canvasRef.current
    const gl = canvas.getContext('webgl2')
    if (!gl) return

    let dpr = Math.max(1, 0.5 * window.devicePixelRatio)
    let program: WebGLProgram | null = null
    let vs: WebGLShader | null = null
    let fs: WebGLShader | null = null
    let buffer: WebGLBuffer | null = null

    // Pointer state
    let active = false
    const pointers = new Map<number, number[]>()
    let lastCoords = [0, 0]
    let moves = [0, 0]

    const vertexSrc = `#version 300 es
precision highp float;
in vec4 position;
void main(){gl_Position=position;}`

    const vertices = [-1, 1, -1, -1, 1, 1, 1, -1]

    const mapCoords = (x: number, y: number) => [x * dpr, canvas.height - y * dpr]

    // Pointer events
    const onPointerDown = (e: PointerEvent) => {
      active = true
      pointers.set(e.pointerId, mapCoords(e.clientX, e.clientY))
    }
    const onPointerUp = (e: PointerEvent) => {
      if (pointers.size === 1) lastCoords = pointers.values().next().value || lastCoords
      pointers.delete(e.pointerId)
      active = pointers.size > 0
    }
    const onPointerLeave = (e: PointerEvent) => {
      if (pointers.size === 1) lastCoords = pointers.values().next().value || lastCoords
      pointers.delete(e.pointerId)
      active = pointers.size > 0
    }
    const onPointerMove = (e: PointerEvent) => {
      if (!active) return
      lastCoords = [e.clientX, e.clientY]
      pointers.set(e.pointerId, mapCoords(e.clientX, e.clientY))
      moves = [moves[0] + e.movementX, moves[1] + e.movementY]
    }

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointerup', onPointerUp)
    canvas.addEventListener('pointerleave', onPointerLeave)
    canvas.addEventListener('pointermove', onPointerMove)

    const compile = (shader: WebGLShader, source: string) => {
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader))
      }
    }

    const setup = () => {
      vs = gl.createShader(gl.VERTEX_SHADER)!
      fs = gl.createShader(gl.FRAGMENT_SHADER)!
      compile(vs, vertexSrc)
      compile(fs, defaultShaderSource)
      program = gl.createProgram()!
      gl.attachShader(program, vs)
      gl.attachShader(program, fs)
      gl.linkProgram(program)
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(program))
      }

      buffer = gl.createBuffer()
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer)
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW)

      const position = gl.getAttribLocation(program, 'position')
      gl.enableVertexAttribArray(position)
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
    }

    const resize = () => {
      dpr = Math.max(1, 0.5 * window.devicePixelRatio)
      canvas.width = window.innerWidth * dpr
      canvas.height = window.innerHeight * dpr
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    const render = (now: number) => {
      if (!program || gl.getProgramParameter(program, gl.DELETE_STATUS)) return

      gl.clearColor(0, 0, 0, 1)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.useProgram(program)
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer)

      const resLoc = gl.getUniformLocation(program, 'resolution')
      const timeLoc = gl.getUniformLocation(program, 'time')
      const moveLoc = gl.getUniformLocation(program, 'move')
      const touchLoc = gl.getUniformLocation(program, 'touch')
      const countLoc = gl.getUniformLocation(program, 'pointerCount')
      const ptrsLoc = gl.getUniformLocation(program, 'pointers')

      gl.uniform2f(resLoc, canvas.width, canvas.height)
      gl.uniform1f(timeLoc, now * 1e-3)
      gl.uniform2f(moveLoc, moves[0], moves[1])

      const first = pointers.size > 0 ? pointers.values().next().value || lastCoords : lastCoords
      gl.uniform2f(touchLoc, first[0], first[1])
      gl.uniform1i(countLoc, pointers.size)

      const coords = pointers.size > 0 ? Array.from(pointers.values()).flat() : [0, 0]
      gl.uniform2fv(ptrsLoc, coords)

      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    }

    const loop = (now: number) => {
      render(now)
      animationFrameRef.current = requestAnimationFrame(loop)
    }

    setup()
    resize()
    loop(0)

    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointerup', onPointerUp)
      canvas.removeEventListener('pointerleave', onPointerLeave)
      canvas.removeEventListener('pointermove', onPointerMove)
      cancelAnimationFrame(animationFrameRef.current)
      if (program && !gl.getProgramParameter(program, gl.DELETE_STATUS)) {
        if (vs) { gl.detachShader(program, vs); gl.deleteShader(vs) }
        if (fs) { gl.detachShader(program, fs); gl.deleteShader(fs) }
        gl.deleteProgram(program)
      }
    }
  }, [])

  return canvasRef
}

export function ShaderCanvas() {
  const canvasRef = useShaderBackground()

  return (
    <canvas
      ref={canvasRef}
      className="fixed top-0 left-0 w-full h-full touch-none"
      style={{ background: 'black' }}
    />
  )
}
