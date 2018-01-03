SELECT TOP 400000
sp.specObjID,sp.objID,sp.ra,sp.dec,sp.z,sp.modelMag_u,sp.modelMag_g,sp.modelMag_r,sp.modelMag_i,sp.modelMag_z,sp.extinction_u,sp.extinction_g,sp.extinction_r,sp.extinction_i,sp.extinction_z
FROM SpecPhotoAll AS sp
WHERE
   sp.objType = 0 AND
   sp.modelMag_r < 17.77 AND
   sp.sciencePrimary > 0 AND
   sp.z < 0.1
ORDER BY sp.specObjID
