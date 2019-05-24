#ifndef	__ICIRC_H_INCLUDED
#define	__ICIRC_H_INCLUDED

/*****************************************************************************/
/* icirc.c								     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* A library for computing intersections of circles.			     */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/* (c) 2011; Pal, A. (apal@szofi.net)					     */
/*****************************************************************************/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

struct circle {
  double	x0, y0;
	double	r;
 };

struct arc {	
  int	cidx;
	double	phi0,dphi;
	int	noidx;
	int	*oidxs;
};

int	icirc_arclist_intersections(circle *circles,int ncircle,arc **rarcs,int *rnarc);
int	icirc_arclist_free(arc *arcs,int narc);

/*****************************************************************************/
/*****************************************************************************/

int icirc_arclist_intersections(circle *circles,int ncircle,arc **routs,int *rnout)
{
 int	i,j;

 arc	*arcs,*aouts;
 int	*acnt;
 int	naout;

 arcs=(arc *)malloc(sizeof(arc)*ncircle*ncircle*2);
 acnt=(int *)malloc(sizeof(int)*ncircle);

 for ( i=0 ; i<ncircle ; i++ )
  {	acnt[i]=0;		}

 for ( i=0 ; i<ncircle ; i++ )
  { for ( j=0 ; j<ncircle ; j++ )
     {	double	xa,ya,xb,yb,ra,rb;
	double	dx,dy,d;
	double	w,phia,phi0;

	if ( i==j )
		continue;

	xa=circles[i].x0;
	ya=circles[i].y0;
	ra=circles[i].r;
	xb=circles[j].x0;
	yb=circles[j].y0;
	rb=circles[j].r;
	dx=xb-xa;
	dy=yb-ya;
	d=sqrt(dx*dx+dy*dy);
	if ( ra+rb <= d )
		continue;
	else if ( d+ra <= rb )
		continue;
	else if ( d+rb <= ra )
		continue;
	w=(ra*ra+d*d-rb*rb)/(2*ra*d);
	if ( ! ( -1.0 <= w && w <= 1.0 ) )
		continue;
	phia=acos(w);
	
	phi0=atan2(dy,dx);
	if ( phi0 < 0.0 )	phi0+=2*M_PI;
		
	if ( acnt[i] <= 0 )
	 {	arc	*a;	
		a=&arcs[2*ncircle*i];
		a[0].phi0=phi0-phia;
		a[0].dphi=2*phia;
		a[1].phi0=phi0+phia;
		a[1].dphi=2*(M_PI-phia);
		acnt[i]=2;
	 }
	else
	 {	arc	*a;	
		double	wp[2],w,dw;
		int	k,n,l;
		wp[0]=phi0-phia;
		wp[1]=phi0+phia;
		a=&arcs[2*ncircle*i];
		n=acnt[i];
		for ( k=0 ; k<2 ; k++ )
		 {	w=wp[k];
			for ( l=0 ; l<n ; l++ )
			 {	dw=w-a[l].phi0;
				while ( dw<0.0 )	dw+=2*M_PI;
				while ( 2*M_PI<=dw )	dw-=2*M_PI;
				if ( dw<a[l].dphi )
				 	break;
			 }
			if ( l<n )
			 {	memmove(a+l+1,a+l,sizeof(arc)*(n-l));
				a[l+1].phi0=a[l].phi0+dw;
				a[l+1].dphi=a[l].dphi-dw;
				a[l].dphi=dw;
				n++;
			 }
		 }
		acnt[i]=n;
	 }

     }
  }

 naout=0;
 for ( i=0 ; i<ncircle ; i++ )
  {	if ( acnt[i] <= 0 )
		naout++;
	else
		naout+=acnt[i];
  }
 aouts=(arc *)malloc(sizeof(arc)*naout);
 j=0;
 for ( i=0 ; i<ncircle ; i++ )
  {	int	k;
	if ( acnt[i] <= 0 )
	 {	aouts[j].cidx=i;
		aouts[j].phi0=0.0;
		aouts[j].dphi=2*M_PI;
		j++;
	 }
	else
	 {	for ( k=0 ; k<acnt[i] ; k++ )
		 {	aouts[j].cidx=i;
			aouts[j].phi0=arcs[2*ncircle*i+k].phi0;
			aouts[j].dphi=arcs[2*ncircle*i+k].dphi;
			j++;
		 }
	 }
  }
 for ( j=0 ; j<naout ; j++ )
  {	double	x,y,dx,dy;
	int	k;

	i=aouts[j].cidx;
	if ( acnt[i] <= 0 )
	 {	x=circles[i].x0+circles[i].r;
		y=circles[i].y0;
	 }
	else
	 {	double	phi;
		phi=aouts[j].phi0+0.5*aouts[j].dphi;
		x=circles[i].x0+circles[i].r*cos(phi);
		y=circles[i].y0+circles[i].r*sin(phi);
	 }
	aouts[j].noidx=0;
	aouts[j].oidxs=NULL;
	for ( k=0 ; k<ncircle ; k++ )
	 {	if ( i==k )	continue;
		dx=x-circles[k].x0;
		dy=y-circles[k].y0;
		if ( dx*dx+dy*dy < circles[k].r*circles[k].r )
		 {	aouts[j].oidxs=(int *)realloc(aouts[j].oidxs,sizeof(int)*(aouts[j].noidx+1));
			*(aouts[j].oidxs+aouts[j].noidx)=k;
			aouts[j].noidx++;
		 }
	 }
  }

 if ( routs != NULL )	*routs=aouts;
 if ( rnout != NULL )	*rnout=naout;

 free(acnt);
 free(arcs);

 return 0;
}

int icirc_arclist_free(arc *arcs,int narc)
{
 int	i;
 for ( i=0 ; i<narc ; i++ )
  {	if ( arcs[i].oidxs != NULL )
		free(arcs[i].oidxs);
  }
 free(arcs);
 
 return 0;
}

#endif
