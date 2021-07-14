#include "Tracer.h"

__device__ color Tracer::trace(const Ray *r, const Scene *sce) const {
  //return color( (r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
  HitRecord ht = sphereTrace(r,sce);

  if (ht.isMiss() || ht.t_ >= max_distance_) // Background
    //return color(0);
    return color((r->dir_.x + 1)+.5, (r->dir_.y + 1)+.5, (r->dir_.z + 1)+.5);
  color c = shade(&ht, sce);
  return c;
};


__device__ HitRecord Tracer::sphereTrace(const Ray *r, const Scene *sce) const {
  if (sce->getShapesNum() <= 0) return HitRecord();

  float t=0;
  float minDistance = infinity;
  float d = infinity;
  ImplicitShape *hit_shape = nullptr;
  while (t < max_distance_) {
    minDistance = infinity; // TODO REMOVEME
    ImplicitShape *shape = sce->getShapes();
    for (int i=0; i < sce->getShapesNum(); i++) {
      d = shape->getDist(r->at(t));
      if (d < minDistance) {
        minDistance = d;
        hit_shape = shape;
      }
      shape++;
    }
    // did we intersect the shape?
    if (minDistance < 0 ||  minDistance <= hit_threshold_ * t) {
      point3 p = r->at(t);
      return HitRecord(r, hit_shape, t, p, hit_shape->getNormalAt(p));
    }
    t += minDistance;
  }
  return HitRecord(t);
}

__device__ color Tracer::shade(const HitRecord *ht, const Scene *sce) const {
  point3 p = ht->p_;
  vec3 v = ht->r_->dir_;
  vec3 n = ht->n_;
  const ImplicitShape *shape = ht->shape_;

  color outRadiance(0);

  vec3 l;
  float nDotl;
  color brdf;

  bool shadow;
  float dist2 = 0;

  // TODO "materials"
  color cdiff = shape->getAlbedo();
  float shininess_factor = shape->getShininess(p);
  color cspec = shape->getSpecular(p);
  ////color cdiff(.5,.3,.8);
  //float shininess_factor = 2;
  //color cspec(0.04);

  if (sce->getLightsNum() > 0) {
    Light *lgt = sce->getLights();

    for (int i=0; i < sce->getLightsNum(); i++) {
      l = (lgt->getPosition() - p);
      dist2 = glm::length(l); // squared dist
      l = glm::normalize(l);
      nDotl = glm::dot(n,l);

      if (nDotl > 0) {
        vec3 r = 2 * nDotl * n - l;
        float vDotr = glm::dot(v, r);
        brdf =
          cdiff / color(M_PI) +
          cspec * powf(vDotr, shininess_factor);

        // With shadows below
        Ray shadowRay(p,l);
        shadow = sphereTraceShadow(&shadowRay, shape, sce);
        //shadow = false; // TODO
        color lightColor = lgt->getColor();
        color lightIntensity = lgt->getIntensity();

        outRadiance += color(1-shadow) * brdf * lightColor * lightIntensity * nDotl
          / (float) (4 * dist2) // with square falloff
          ;
      }
      lgt++;
    }
  }
  if (sce->hasAmbientLight()) {
    AmbientLight* ambientLight = sce->getAmbientLight();
    outRadiance += ambientLight->getColor() * ambientLight->getIntensity() * cdiff;
  }
  return glm::clamp(outRadiance, color(0,0,0), color(1,1,1));
}


__device__ bool Tracer::sphereTraceShadow(const Ray *r, const ImplicitShape *shapeToShadow, const Scene *sce) const {
  if (sce->getShapesNum() <= 0) return false;

  float t=0;
  float minDistance = infinity;
  float d = infinity;
  point3 from = r->at(t);
  while (t < max_distance_) {
    minDistance = infinity; // TODO REMOVEME
    ImplicitShape *shape = sce->getShapes();
    for (int i=0; i < sce->getShapesNum(); i++) {
      from = r->at(t);
      d = shape->getDist(from);

      // Self-Hit Shadowing Error Solution
      if (shape == shapeToShadow && d <= hit_threshold_ * t) {
        // move "from" a bit over the surface (along the normal direction)
        d = shape->getDist(
            from + shape->getNormalAt(from) * vec3(10e-7)
            );
      }

      if (d < minDistance) {
        minDistance = d;
        if (minDistance < 0 ||  minDistance <= hit_threshold_ * t) {
          return true;
        }
      }
      shape++;
    }
    t += minDistance;
  }
  return false;
}
