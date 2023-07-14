import requests

capture_title = "test_sofa1"
auth_headers = {'Authorization': 'luma-api-key=900b4468-3b1c-4d9b-b631-c2535c6722e3-b5182a6-1d25-43bb-b815-b67cb48e2f71'}

#response = requests.post("https://webapp.engineeringlumalabs.com/api/v2/capture",
#                         headers=auth_headers,
#                         data={'title': capture_title})
#capture_data = response.json()
#upload_url = capture_data['signedUrls']['source']
#slug = capture_data['capture']['slug']
#
#print(capture_data)
#print("Upload URL:", upload_url)
#print("Capture slug:", slug)
#
#with open("sofa.mp4", "rb") as f:
#    payload = f.read()
#
## upload_url from step (1)
#response = requests.put(upload_url, headers={'Content-Type': 'text/plain'}, data=payload)
#
## Note: the payload should be bytes containing the file contents (as shown above)!
## A common pitfall is uploading the file name as the file contents
#
#print(response.text)
#
# slug from Capture step
#
#auth_headers = {'Authorization': 'luma-api-key=900b4468-3b1c-4d9b-b631-c2535c6722e3-b5182a6-1d25-43bb-b815-b67cb48e2f71'}
#response = requests.post(f"https://webapp.engineeringlumalabs.com/api/v2/capture/{slug}",
#                         headers=auth_headers)
#
#print(response.text)

# slug from Capture step

auth_headers = {'Authorization': 'luma-api-key=900b4468-3b1c-4d9b-b631-c2535c6722e3-b5182a6-1d25-43bb-b815-b67cb48e2f71'}
response = requests.get(f"https://webapp.engineeringlumalabs.com/api/v2/capture/stirringly-refreshing-3j-744718",
                        headers=auth_headers)

print(response.text)

#{"title":"test_sofa1","type":"reconstruction","location":null,"privacy":"private","date":"2023-05-04T04:24:18.000Z","username":null,"status":"complete","slug":"stirringly-refreshing-3j-744718","editUrl":"https://captures.lumalabs.ai/editor/stirringly-refreshing-3j-744718","slugUrl":"https://captures.lumalabs.ai/stirringly-refreshing-3j-744718","embedUrl":"https://captures.lumalabs.ai/embed/stirringly-refreshing-3j-744718?mode=slf&background=%23ffffff&color=%23000000&showTitle=true&loadBg=true&logoPosition=bottom-left&infoPosition=bottom-right&cinematicVideo=undefined&showMenu=false","latestRun":{"status":"finished","progress":100,"currentStage":"Done","artifacts":[{"url":"https://cdn-luma.com/089307e90dd9262662f4ccdbb7bd23bea198e47a8d399c0dab59ac12142014ca.jpg","type":"thumb"},{"url":"https://cdn-luma.com/8cf27d7e8dae02b6eee711eb40e93702f8a281b5bdd4af505c8c5bba8103535a.jpg","type":"preview_360"},{"url":"https://cdn-luma.com/cd93df6d0aa7821a6d1cf5f2ee31ce44392476a72656b66df64dad4204aadd80.mp4","type":"video_with_background"},{"url":"https://cdn-luma.com/d7399abc3bbbe415c66e7ed72494bf2ca9940d13d1e23bbe80427b6b0a874ed7.jpg","type":"video_with_background_preview"},{"url":"https://cdn-luma.com/91945c788411ebe6f0c512d6485dd4d6f4b658db245ba16fa761830d5493298d.ply","type":"full_mesh","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/8a508ddd55136259f948e9ea62fc8aea619abde04f2ed9ffbc2cfa95a538a0be.ply","type":"point_cloud","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/0111b7544b0cdcb0c94bcdc34376c04771f4ce037b4dbf2469bc0ff5bcec7250.glb","type":"textured_mesh_lowpoly_glb","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/b4d518b97fc838b5235439b6ef326b220530c52f7968ecd193610eb5ef533017.glb","type":"textured_mesh_medpoly_glb","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/97e3f3b0a3dd177da4b369b0cd5faf807018d5c4709d434d171b0a8f5ad92d40.glb","type":"textured_mesh_glb","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/eff9bb75a0146184cddfc7c51c17c78409814a063eb1bb8151d9a68e52666e77.zip","type":"textured_mesh_lowpoly_obj","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/997a6cf9f6c742b2957740e2d453d67588da6fc7818d31d312159ab9772ce862.zip","type":"textured_mesh_medpoly_obj","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/dfc595b363a80136f7576950b690ca131ecf2f3c478296b4584002fd9d83672a.zip","type":"textured_mesh_obj","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/cd155edfe045677c8ef31f8b87cae323628c3a68f1c910c36f33e4dfd7626f57.usdz","type":"textured_mesh_lowpoly_usdz","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/b6537868fe881207698fb56546bc98d0e9d6079ac9d34fed7ba5f072f3cda9ae.usdz","type":"textured_mesh_medpoly_usdz","scale_to_world":0.975387434159297},{"url":"https://cdn-luma.com/b35931780f7a5960885a377f3796eb8e110796403f9e0d82fbbd5a79de6b821b.usdz","type":"textured_mesh_usdz","scale_to_world":0.975387434159297}]}}