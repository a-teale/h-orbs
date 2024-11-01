import numpy, dash
import scipy.special
import plotly.graph_objects as go
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

BoxSizes = [7.5,7.5,15.0,25.0,35.0,50.0]
IsoValue = [0.024,0.012,0.006,0.003,0.002]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div([
    html.H1('Hydrogen Atomic Orbitals',style={'font-family': 'sans-serif',"margin-top": "20","margin-bottom": "10"},className='page_header'),
    html.Div([
        html.Div(html.H4('Orbital Quantum Numbers',style={'font-family': 'sans-serif'}),style={'width': '60%', 'display': 'inline-block'})]),
        #html.Div(html.H4('Gaussian Exponent',style={'font-family': 'sans-serif'}),style={'width': '40%', 'display': 'inline-block'},id='gau_exp_h4')]),
    html.Div([
        html.Div([html.Label('Principal Quantum Number',style={'font-family': 'sans-serif'}),dcc.Input(id="hydrogen_n",type="number",placeholder="principal quantum number",min=1,max=5,step=1,value=1,style={'font-family': 'sans-serif','justify': 'center'})],style={'width': '20%', 'display': 'inline-block'}),
        html.Div([html.Label('Orbital Angular Momentum',style={'font-family': 'sans-serif'}),dcc.Input(id="hydrogen_l",type="number",placeholder="orbital angular momentum",min=0,max=4,step=1,value=0,style={'font-family': 'sans-serif','justify': 'center'})],style={'width': '20%', 'display': 'inline-block'}),
        html.Div([html.Label('Magnetic Quantum Number',style={'font-family': 'sans-serif'}),dcc.Input(id="hydrogen_m",type="number",placeholder="magnetic quantum number",min=-4,max=4,step=1,value=0,style={'font-family': 'sans-serif','justify': 'center'})],style={'width': '20%', 'display': 'inline-block'}),
        #html.Div(dcc.Slider(id='gaussexp',min=-3.0,max=3.0,step=0.01,value=0.0,marks={i: '{}'.format(10**i) for i in range(-3,4,1)},updatemode='mouseup',tooltip=dict(always_visible=False,placement='bottom'),dots=False),style={'width': '40%', 'display': 'inline-block'})
        ],style={'width': '100%', 'display': 'inline-block','vertical-align': 'middle'}),
        html.Br(),html.Br(),
        html.Div([
            html.Div([html.H4('Radial Wavefunction',style={'font-family': 'sans-serif',"margin-top": "10","margin-bottom": "10",'textAlign': 'center'}),dcc.Graph(id='rdf',config={'displayModeBar': False})],style={'width': '30%', 'display': 'inline-block'}),
            html.Div([html.H4('Orbital IsoSurface',style={'font-family': 'sans-serif',"margin-top": "10","margin-bottom": "10",'textAlign': 'center'}),dcc.Graph(id='hydrogen_iso',config={'displayModeBar': False})],style={'width': '35%', 'display': 'inline-block'}),
            #html.Div([html.H4('Gaussian IsoSurface',style={'font-family': 'sans-serif',"margin-top": "10","margin-bottom": "10",'textAlign': 'center'}),dcc.Graph(id='gauss_iso',config={'displayModeBar': False})],style={'width': '35%', 'display': 'inline-block'})
        ],style={'width': '100%', 'display': 'inline-block','vertical-align': 'middle'})
])

#@app.callback(
#     Output('gau_exp_h4','children'),
#    [Input('gaussexp', 'value')])
#def UpdateGexp(s):
#    e = pow(10.0,s)
#    return html.H4('Gaussian Exponent: {0:.3f}'.format(e),style={'font-family': 'sans-serif'})

@app.callback(
    [Output('hydrogen_l','max'),
     Output('hydrogen_l','value')],
    [Input('hydrogen_n', 'value')])
def UpdateLMax(n):
    return n-1, 0

@app.callback(
    [Output('hydrogen_m','min'),
     Output('hydrogen_m','max'),
     Output('hydrogen_m','value')],
    [Input('hydrogen_l', 'value')])
def UpdateMRng(l):
    return -l, l, 0

@app.callback(
    Output('rdf', 'figure'),
    [Input('hydrogen_n', 'value'),
     Input('hydrogen_l', 'value')])
     #Input('gaussexp'  , 'value')])
def UpdateRDF(n,l):
    r = numpy.linspace(0,BoxSizes[n],num=101)
    #e = pow(10.0,s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r,y=HydrogenRadial(n,l,r),mode='lines',line_shape='spline',name='Atomic Orbital'))
    #fig.add_trace(go.Scatter(x=r,y=GaussianRadial(l,e,r),mode='lines',line_shape='spline',name='Gaussian Function'))
    fig.update_layout(template='plotly_white',margin={'t': 0, 'l': 0, 'r': 0, 'b': 0},hovermode='closest',autosize=True,legend=dict(orientation='h',yanchor='middle',y=1.0,xanchor='center',x=0.5,bgcolor="rgba(0,0,0,0)",font=dict(size=10)),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(title='r',tickformat = '.1f',showgrid=False,showline=True,linewidth=2,linecolor='black',ticks="outside",tickwidth=2,ticklen=6,title_font=dict(size=12),tickfont=dict(size=8)),yaxis=dict(title='R(r)',tickformat = '.2f',showgrid=False,showline=True,linewidth=2,linecolor='black',ticks="outside",tickwidth=2,ticklen=6,title_font=dict(size=12),tickfont=dict(size=8))) 
    return fig

@app.callback(
    Output('hydrogen_iso','figure'),
    [Input('hydrogen_n'  , 'value'),
     Input('hydrogen_l'  , 'value'),
     Input('hydrogen_m'  , 'value')])
def UpdateOrbIso(n,l,m):
    rmax = BoxSizes[n]
    npts = 51
    u = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
    v = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
    w = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
    x, y, z = numpy.meshgrid(u,v,w)
    Orb = HydrogenOrbital(n,l,m,x,y,z)
    fig = go.Figure()
    fig.add_trace(Atom())
    Iso, Min, Max = IsoSurface(x,y,z,Orb,IsoValue[l],'Picnic',0.3,max(n-l,2),True)
    fig.add_trace(Iso)
    fig.update_layout(GetLayout(None))
    fig.update_layout(GetRange(Min,Max,n))
    return fig

#@app.callback(
#    Output('gauss_iso'   ,'figure'),
#    [Input('hydrogen_n'  , 'value'),
#     Input('hydrogen_l'  , 'value'),
#     Input('hydrogen_m'  , 'value'),
#     Input('gaussexp'    , 'value')])
#def UpdateGauIso(n,l,m,s):
#    e = pow(10.0,s)
#    rmax = BoxSizes[n]
#    npts = 51
#    u = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
#    v = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
#    w = numpy.linspace(-rmax,rmax,npts) + 1.0e-8
#    x, y, z = numpy.meshgrid(u,v,w)
#    Orb = GaussianOrbital(l,m,e,x,y,z)
#    fig = go.Figure()
#    fig.add_trace(Atom())
#    Iso, Min, Max = IsoSurface(x,y,z,Orb,IsoValue[l],'Picnic',0.3,max(n-l,2),True)
#    fig.add_trace(Iso)
#    fig.update_layout(GetLayout(None))
#    fig.update_layout(GetRange(Min,Max,n))
#    return fig

def IsoSurface(x,y,z,val,iso=0.08,colorscale='Picnic',opacity=0.3,nsurf=2,visible=True):
    trace = go.Isosurface(x = x.flatten(),
                          y = y.flatten(), 
                          z = z.flatten(), 
                          value = val.flatten(),
                          surface_count = nsurf,
                          colorscale = colorscale,
                          visible = visible,
                          showscale = False,
                          isomin = -1 * iso,
                          isomax =  1 * iso, 
                          flatshading = False,
                          lighting = SurfaceStyles['matte'], 
                          caps = dict(x_show=False,y_show=False,z_show=False),
                          opacity = opacity)
                          
    max_range_x = numpy.max(x.flatten())
    max_range_y = numpy.max(y.flatten())
    max_range_z = numpy.max(z.flatten())
    min_range_x = numpy.min(x.flatten())
    min_range_y = numpy.min(y.flatten())
    min_range_z = numpy.min(z.flatten())
    max_range = max(max_range_x,max_range_y,max_range_z)
    min_range = min(min_range_x,min_range_y,min_range_z)
    
    return trace, min_range, max_range

def SphericalHarmonic(l,m,t,p):
    a = (2.0*l+1.0) * scipy.special.factorial(l-m)
    b = 4.0 * numpy.pi * scipy.special.factorial(l+m)
    c = scipy.special.lpmv(m,l,numpy.cos(t))
    d = numpy.exp(1.0j*m*p)
    Ylm = numpy.sqrt(a/b) * c * d
    return Ylm
    
def RealSH(l,m,t,p):
    if   m > 0:
        return numpy.sqrt(2) * pow(-1.0,m) * SphericalHarmonic(l,abs(m),t,p).real
    elif m < 0:
        return numpy.sqrt(2) * pow(-1.0,m) * SphericalHarmonic(l,abs(m),t,p).imag
    else:
        return SphericalHarmonic(l,m,t,p)
    
def HydrogenRadial(n,l,r):
    a = pow(2.0/n,l+1.5)
    b = numpy.sqrt(scipy.special.factorial(n-l-1)/(2.0*n*scipy.special.factorial(n+l)))
    c = pow(r,l)
    d = scipy.special.eval_genlaguerre(n-l-1,l+l+1,2.0*r/n)
    e = numpy.exp(-r/n)
    Rnl = a * b * c * d * e
    return Rnl

def SlaterRadial(l,s,r):
    a = pow(2.0*s,1.5)
    b = 1.0 / numpy.sqrt(scipy.special.factorial(l+l+2))
    c = pow(2.0*s*r,l)
    d = numpy.exp(-s*r)
    Rl = a * b * c * d
    return Rl

def GaussianRadial(l,s,r):
    a = 2.0 * pow(2.0*s,0.75) / pow(numpy.pi,0.25)
    b = numpy.sqrt(pow(2.0,l)/scipy.special.factorial2(l+l+1))
    c = pow(numpy.sqrt(2.0*s)*r,l)
    d = numpy.exp(-s*r*r)
    Rl = a * b * c * d
    return Rl

def HydrogenOrbital(n,l,m,x,y,z):
    r, t, p = Car2Sph(x,y,z)
    return (HydrogenRadial(n,l,r) * RealSH(l,m,t,p)).real

def SlaterOrbital(l,m,s,x,y,z):
    r, t, p = Car2Sph(x,y,z)
    return (SlaterRadial(l,s,r) * RealSH(l,m,t,p)).real

def GaussianOrbital(l,m,s,x,y,z):
    r, t, p = Car2Sph(x,y,z)
    return (GaussianRadial(l,s,r) * RealSH(l,m,t,p)).real

def RealAngular(l,m,x,y,z):
    r, t, p = Car2Sph(x,y,z)
    return RealSH(l,m,t,p).real

def Sph2Car(r,t,p):
    x = r * numpy.cos(t) * numpy.sin(p)
    y = r * numpy.sin(t) * numpy.sin(p)
    z = r * numpy.cos(p)
    return x, y, z

def Car2Sph(x,y,z):
    xy = x*x + y*y
    r = numpy.sqrt(xy+z*z)
    t = numpy.arctan2(numpy.sqrt(xy),z)
    p = numpy.arctan2(y,x)
    p[p < 0] += 2.0*numpy.pi
    return r, t, p

def Atom(r=0.3,n=5):
    phi   = numpy.linspace(0,2.0*numpy.pi,2*n)
    theta = numpy.linspace(-0.5*numpy.pi,0.5*numpy.pi,n)
    phi, theta = numpy.meshgrid(phi[1:],theta)
    pts = numpy.zeros((3,phi.size))
    pts[0] = r * (numpy.cos(theta) * numpy.sin(phi)).flatten()
    pts[1] = r * (numpy.cos(theta) * numpy.cos(phi)).flatten()
    pts[2] = r * (numpy.sin(theta)).flatten()
    iAtm = go.Mesh3d({'x': pts[0],
                      'y': pts[1],
                      'z': pts[2],
                      'color': "slategrey",
                      'opacity': 0.7,
                      'alphahull': 0,
                      'flatshading': False,
                      'cmin': -7,
                      'lighting': SurfaceStyles['matte'],
                      'lightposition': {'x': 100, 'y': 200, 'z': 0}})
    return iAtm

Ang2Bohr = 1.889725989
Bohr2Ang = 0.529177249
Amu2SiKg = 1.660539040e-27

SurfaceStyles = {'matte': {'ambient'             : 0.60,
                           'diffuse'             : 0.35,
                           'fresnel'             : 0.05,
                           'specular'            : 0.03,
                           'roughness'           : 0.05,
                           'facenormalsepsilon'  : 1e-15,
                           'vertexnormalsepsilon': 1e-15}}

def GetRange(min_range,max_range,n=1,overage=2.5):
    axis = {'autorange': False, 'range': (min_range*overage,max_range*overage)}

    layout = {'scene': {'xaxis': axis, 'yaxis': axis, 'zaxis': axis},
              'scene_camera': {'up': {'x': 0, 'y': 0, 'z': 1}, 'center': {'x': 0, 'y': 0, 'z': 0},
                               'eye': {'x': n*2.2/max_range, 'y': n*2.2/max_range, 'z': n*2.2/max_range}}}
    return layout

def GetLayout(figsize=None):
    axis = {'showgrid'      : False,
            'zeroline'      : False,
            'showline'      : False,
            'title'         :    '',
            'ticks'         :    '',
            'showticklabels': False,
            'showbackground': False,
            'showspikes'    : False}

    layout = {'scene_aspectmode'          : 'manual',
              'scene_aspectratio'         : {'x': 1, 'y': 1, 'z': 1},
              'scene_xaxis_showticklabels': False,
              'scene_yaxis_showticklabels': False,
              'scene_zaxis_showticklabels': False,
              'dragmode'                  : 'orbit',
              'template'                  : 'plotly_white',
              'showlegend'                : False,
              'hovermode'                 : False,
              'margin'                    : {'t': 0, 'l': 0, 'b': 0, 'r': 0},
              'scene'                     : {'xaxis': axis, 'yaxis': axis, 'zaxis': axis}}

    if figsize is not None:
        layout['height'] = figsize[0]
        layout['width']  = figsize[1]

    return layout

if __name__ == '__main__':
    app.run_server(debug=False)
