### SERVICE: Great circle

This service allows the operator to manipulate great circles (abbreviated GCI).

### MMI Disposition

![Screenshot of the MAC (Great Circle) menu interface. The menu has tabs for N°T, Vel, Radio, and Time res. Below the tabs is a large empty area. At the bottom, there are input fields for NT, Vel.MAC, Sec.vel.MAC, Hora Activación, and TiempFMAC, along with buttons for OK, Aplicar, Reinc, Cancelar, and Ayuda. To the right of the main menu is a small icon showing a ship's silhouette within a circle, with the label CMI below it.](7e8f8ba045fdae1a5c7bff53622473af_4_img.webp)

### Operator actions

<table border="1">
<thead>
<tr>
<th>Operator action</th>
<th>Logical entry</th>
<th>Real entry</th>
<th>Related departure</th>
<th>Restrictions</th>
</tr>
</thead>
<tbody>
<tr>
<td>Create SOI</td>
<td>CMI track</td>
<td>Enter "NT", "Vel.MAC" and/or "Sec.vel.MAC" and/or "Hora activación" and/or "TiempFMAC"</td>
<td>CMI<br/>CMI Report</td>
<td>Only for lost surface point tracks and reference positions Maximum number of CMI =10</td>
</tr>
<tr>
<td>Eliminate the CMI</td>
<td>CMI track</td>
<td>Select "MAC" and press "Remove"</td>
<td>CMI deleted Report<br/>CMI deleted</td>
<td>Only for manually initiated tracks</td>
</tr>
<tr>
<td>Change CMI</td>
<td>CMI track and CMI parameters</td>
<td>Select "MAC" and enter "Vel.MAC" and/or "Sec.vel.MAC" and/or "Hora activación" and/or "TiempFMAC"</td>
<td>CMI<br/>CMI Report</td>
<td>-</td>
</tr>
</tbody>
</table>

### System reports

<table border="1">
<thead>
<tr>
<th>System report</th>
<th>Restrictions</th>
</tr>
</thead>
<tbody>
<tr>
<td>CMI Report</td>
<td>-</td>
</tr>
<tr>
<td>CMI track</td>
<td>-</td>
</tr>
<tr>
<td>CMI Parameter</td>
<td>-</td>
</tr>
</tbody>
</table><table border="1">
<thead>
<tr>
<th>Acción del operador</th>
<th>Entrada lógica</th>
<th>Entrada real</th>
<th>Salida relacionada</th>
<th>Restricciones</th>
</tr>
</thead>
<tbody>
<tr>
<td>Imprimir Definiciones de tarea técnica interna</td>
<td>-</td>
<td>Pulse Impr.</td>
<td>Aparece la ventana "Imprimir" que permite imprimir las Definiciones de tareas técnicas internas Ver Archivo y Recuperación de servicio</td>
<td>El botón imprimir no está disponible (ajustado por parámetro del sistema<sup>d</sup>: Nombre de grupo = LSSCONF, Nombre de elemento = Impresión_habilitada)</td>
</tr>
</tbody>
</table>

1. Si ya existe una definición de posición de botón de conjunto de trabajo nueva, la posición de todas las definiciones de posición de botón de conjunto de trabajo con el mismo nombre de función y con una posición igual o superior a la nueva (incluidas las posiciones "adjuntas vacías"), aumentará en uno. En caso de que exista una definición de botón de conjunto de trabajo para la última posición posible, aparecerá una ventana de error con el mensaje 1.
2. Las definiciones de posición de botón del conjunto de trabajo de la posición nueva y anterior se intercambiarán, también si la posición nueva y anterior es una posición "ajunta vacía".
3. El borrado de una posición de botón de conjunto de trabajo genera una posición "ajunta vacía" si ya existe una definición de botón de conjunto de trabajo con una posición mayor. Si se borra una posición "ajunta vacía", la posición de todas las definiciones de posición de botón de conjunto de trabajo con el mismo nombre de función y con una posición superior a la borrada (incluidas las posiciones "adjuntas vacías"), disminuirá en una.
4. Un cambio de este parámetro sólo surtirá efecto después del arranque del servicio.

### Informes del sistema

<table border="1">
<thead>
<tr>
<th>Informe de sistema</th>
<th>Restricciones</th>
</tr>
</thead>
<tbody>
<tr>
<td>Definición del botón de conjunto de trabajo</td>
<td>-</td>
</tr>
<tr>
<td>Nombre de rol</td>
<td>-</td>
</tr>
<tr>
<td>Nombre de tarea operativa</td>
<td>-</td>
</tr>
<tr>
<td>Ventana de mensaje de error 1</td>
<td>-</td>
</tr>
</tbody>
</table>

### Definiciones de datos

<table border="1">
<thead>
<tr>
<th>Elemento</th>
<th>Definiciones de valores y datos</th>
</tr>
</thead>
<tbody>
<tr>
<td>Definición del botón de conjunto de trabajo</td>
<td>=( Nombre de función, [Nombre de tarea operativa], Indicación de posición)<br/>Nota:<br/>Una posición "ajunta vacía" (posición para la que no está definida ninguna tarea operativa, pero para la que existe una definición de botón de conjunto de trabajo con una posición superior) se visualiza con un nombre de función y posición, pero sin un nombre de tarea operativa.</td>
</tr>
<tr>
<td>Nombre de rol</td>
<td>Cadena, máx. 32 caracteres</td>
</tr>
<tr>
<td>Nombre de tarea operativa</td>
<td>Cadena, máx. 32 caracteres</td>
</tr>
<tr>
<td>Indicación de posición</td>
<td>[1, 99]; 1; -</td>
</tr>
<tr>
<td>Ventana de mensaje de error 1</td>
<td>["La inserción no ha sido posible porque la posición especificada y la última posible están ocupadas"]</td>
</tr>
</tbody>
</table>